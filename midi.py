#!/usr/bin/python3


import os
import threading
import time
import logging
import mido
import numpy as np
import _rpi_ws281x as ws
import pychord


logging.basicConfig(filename='midi.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# MIDI Port
# aconnect -l コマンドで確認

#MIDI_PORT = 'rtpmidi rp4midi:Network 128:0'
MIDI_PORT = 'UM-ONE:UM-ONE MIDI 1'

# 
LED_MANUAL = 144
MANUAL_COUNT = 4  # 0: Upper, 1: Lower, 2: Pedal, 3: Exp

# LED strip configuration:
LED_COUNT = LED_MANUAL*MANUAL_COUNT // 2 # Number of LED pixels.
# 利用可能なGPIOピンはこの辺を参考にする
# https://github.com/jgarff/rpi_ws281x/blob/9be313f77aa494036e2dc205b6ec2860e7ee988c/pwm.c#L38
LED1_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM0).
LED2_PIN = 13          # GPIO pin connected to the pixels (18 uses PWM1).
LED_FREQ_HZ = 800000   # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10           # DMA channel to use for generating signal (try 10)
LED1_BRIGHTNESS = 128   # Set to 0 for darkest and 255 for brightest
LED2_BRIGHTNESS = 192  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False     # True to invert the signal (when using NPN transistor level shift)

leds = ws.new_ws2811_t()

channel0 = ws.ws2811_channel_get(leds, 0) # Upper/Lower
channel1 = ws.ws2811_channel_get(leds, 1) # Pedal/Exp

ws.ws2811_channel_t_count_set(channel0, LED_COUNT)
ws.ws2811_channel_t_gpionum_set(channel0, LED1_PIN)
ws.ws2811_channel_t_invert_set(channel0, LED_INVERT)
ws.ws2811_channel_t_brightness_set(channel0, LED1_BRIGHTNESS)
ws.ws2811_channel_t_strip_type_set(channel0, ws.WS2812_STRIP)

ws.ws2811_channel_t_count_set(channel1, LED_COUNT)
ws.ws2811_channel_t_gpionum_set(channel1, LED2_PIN)
ws.ws2811_channel_t_invert_set(channel1, LED_INVERT)
ws.ws2811_channel_t_brightness_set(channel1, LED2_BRIGHTNESS)
ws.ws2811_channel_t_strip_type_set(channel1, ws.WS2812_STRIP)

ws.ws2811_t_freq_set(leds, LED_FREQ_HZ)
ws.ws2811_t_dmanum_set(leds, LED_DMA)

resp = ws.ws2811_init(leds)
if resp != ws.WS2811_SUCCESS:
	message = ws.ws2811_get_return_t_str(resp)
	raise RuntimeError('ws2811_init failed with code {0} ({1})'.format(resp, message))


# led_array: <LED_MARGIN> <LED_MANUAL> <LED_MARGIN>
#                        |<---------->|
#                           neopixel  この範囲をneopixelで表示

LED_MARGIN = 144


#

perf_led = []
perf_led_intr = []
perf_led_lasttime = 0
perf_term = []
perf_term_intr = []
perf_term_lasttime = 0
perf_midi = []
perf_midi_intr = []
perf_midi_lasttime = 0

# 現在のモード番号
current_mode = 0
mode_name = {
	0: 'Normal',
	1: 'AKEBOSHI',
}
# 現在のシーン番号
current_scene = 0
# クロック
midi_clock = 0
# 演奏開始時間
music_start_time = 0
music_start_clock = 0 # 演奏中のみ増加
music_stop = 0
scene_start_time = 0
scene_note_count = [0] * 16
beat_in_a_bar = 4
# MIDI Event count
midi_event_count = 0
# note_on中のdict。(channel, note) -> (scene, time)
note_state = {}
# 直近のnote_on/offのリスト。(type, channel, note, scene, time)。typeはon/off
note_event = []
# Expression
current_expression = 0

# event保持時間
NOTE_EVENT_TTL = 5
#NOTE_EVENT_TTL = 60


#### Pattern

# color = [R, G, B]

note_color = np.array([
	[202,  21, 116], # C
	[ 15, 169, 109], # C#
	[218,  85,  40], # D
	[ 28, 114, 165], # D#
	[238, 236,  44], # E
	[124,  43, 122], # F
	[103, 187,  65], # F#
	[211,  38,  58], # G
	[ 13, 165, 166], # G#
	[236, 152,  40], # A
	[ 75,  68, 138], # A#
	[187, 219,  46], # B
]).astype(np.uint32)

led_note_shape = np.array([
	[102, 102, 255], [102, 102, 255], [102, 102, 255],
	[255, 255, 255], [255, 255, 255], [255, 255, 255],
	[102, 102, 255], [102, 102, 255], [102, 102, 255],
]).astype(np.uint32)

led_note_particle_shape = np.array([
	[102, 102, 255]
]).astype(np.uint32)

led_note_shape_white = np.array([
	[255, 255, 255], [255, 255, 255], [255, 255, 255],
]).astype(np.uint32)

led_note_shape_red = np.array([
	[255,   0,   0], [255,   0,   0], [255,   0,   0],
]).astype(np.uint32)

led_note_shape_blue = np.array([
	[  0,   0, 255], [  0,   0, 255], [  0,   0, 255],
]).astype(np.uint32)

led_note_particle_shape_off = np.array([
	[  0,   0,  30]
]).astype(np.uint32)

## Scene change

# channelごとに10音まで記録するので、trigger_notesは最大10音まで
MAX_NOTE_HISTORY = 10

# note_type='on' なら、trigger発動したノート自体から新しいscene
# note_type='off' なら、trigger発動したノート自体は以前のscene
scene_trigger = {
	1: { # AKEBOSHI
		# (current_scene, note_type, channel) -> [ (trigger_notes, next_scene) ]
		# channel 0:Solo, 1:Pedal, 2:Lower, 3:Upper
		(0, 'note_on', 2): [
			(['F#4'], 1),
		],
		# 2小節目
		(1, 'note_on', 0): [
			(['B4'], 2),
		],
		# 3小節目
		(2, 'note_on', 3): [
			(['B4'], 3),
		],
		# 4小節目[2]
		(3, 'note_on', 3): [
			(['D#4', 'A4'], 4),
			(['D#4', 'B4', 'A4'], 4),
		],
		# 12小節目[3]
		(4, 'note_on', 2): [
			(['D4'], 5),
		],
		# 17小節目
		(5, 'note_on', 2): [
			(['G4', 'B3'], 6),
			(['G4', 'C#4'], 6),
			(['G4', 'F#4'], 6),
		],
		# 17小節目[4]
		(6, 'note_on', 3): [
			(['D4'], 7),
		],
		# 26小節目
		(7, 'note_on', 3): [
			(['A#4', 'A4', 'G4', 'G4', 'F4'], 8),
		],
		# 32小節目
		(8, 'note_on', 3): [
			(['A4', 'G4', 'F4', 'G4'], 9),
		],
		# 32小節目
		(9, 'note_on', 3): [
			(['G4', 'F4', 'A4'], 10),
			(['G4', 'A4', 'F4'], 10),
		],
		# 35小節目[5]
		(10, 'note_on', 3): [
			(['D4'], 11),
		],
		# 35小節目
		(11, 'note_on', 3): [
			(['F4', 'G4', 'E5'], 12),
		],
	}
}

# channel -> manual (0:upper, 1:lower, 2:pedal)
channel_map = {
	0: 1,
	1: 2,
	2: 1,
	3: 0,
	4: 1,
	5: 1,
	6: 1,
	7: 1,
	8: 1,
	9: 1,
	10: 1,
	11: 1,
	12: 1,
	13: 1,
	14: 1,
	15: 1,
}


#### Util

def ch_to_name(ch):
	name = ['S', 'P', 'L', 'U', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
	
	return name[ch]

def note_to_name(no):
	name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	
	return name[no % 12] + str(no//12 - 1)

note_name_to_no = {}
for no in range(128):
	note_name_to_no[note_to_name(no)] = no

def color_interp(c0, c1, x0, x1, p):
	return np.array([
		np.interp(p, [x0, x1], [c0[0], c1[0]]),
		np.interp(p, [x0, x1], [c0[1], c1[1]]),
		np.interp(p, [x0, x1], [c0[2], c1[2]]),
	], dtype=np.uint32)

def color_interp_period(c0, c1, x0, x1, p, period):
	return np.array([
		np.interp(p, [x0, x1], [c0[0], c1[0]], period=period),
		np.interp(p, [x0, x1], [c0[1], c1[1]], period=period),
		np.interp(p, [x0, x1], [c0[2], c1[2]], period=period),
	], dtype=np.uint32)


#### LED Thread



led_array = np.zeros((0, 3), dtype=np.uint32)

def init_led_array():
	global led_array
	led_array = np.zeros((MANUAL_COUNT, (LED_MARGIN + LED_MANUAL + LED_MARGIN), 3), dtype=np.uint32)

def draw_led_array(manual, idx, shape):
	global led_array
	array_len = len(shape)
	led_idx = LED_MARGIN + idx - (array_len - 1) // 2
	led_array[manual, led_idx : led_idx+array_len] += shape

def draw_led_array_brightness(manual, idx, shape, brightness):
	if brightness < 0:
		return
	global led_array
	array_len = len(shape)
	led_idx = LED_MARGIN + idx - (array_len - 1) // 2
	led_array[manual, led_idx : led_idx+array_len] += (shape.astype(np.float32) * brightness).astype(np.uint32)

def draw_led_array_bg(manual, color, brightness):
	global led_array
	led_array[manual, :] += np.multiply(color, brightness).astype(np.uint32)

def show_led():
	global led_array
	
	pixel_array = np.concatenate([
		led_array[0:1, LED_MARGIN:LED_MARGIN+LED_MANUAL][0],
		led_array[1:2, LED_MARGIN:LED_MARGIN+LED_MANUAL][0][::-1],
		led_array[2:3, LED_MARGIN:LED_MARGIN+LED_MANUAL][0],
		led_array[3:4, LED_MARGIN:LED_MARGIN+LED_MANUAL][0],
	])
	pixel_array = (
		(pixel_array[:, 0:1].flatten() << 16) +
		(pixel_array[:, 1:2].flatten() << 8) +
		pixel_array[:, 2:3].flatten()
	)
	pixel_array1 = pixel_array[LED_MANUAL*2-1::-1].tolist()
	pixel_array2 = pixel_array[:LED_MANUAL*2-1:-1].tolist()
	
#	logger.debug("pixel1:{}".format(len(pixel_array1)))
#	logger.debug("pixel2:{}".format(len(pixel_array2)))
	
	for i, x in enumerate(pixel_array1):
		ws.ws2811_led_set(channel0, i, x)
	
	for i, x in enumerate(pixel_array2):
		ws.ws2811_led_set(channel1, i, x)
	
	resp = ws.ws2811_render(leds)
	if resp != ws.WS2811_SUCCESS:
		message = ws.ws2811_get_return_t_str(resp)
		raise RuntimeError('ws2811_render failed with code {0} ({1})'.format(resp, message))


def note_to_index(manual, note_no):
	# UPPER: [ 3: 48]  [ 3:103] 
	# LOWER: [ 2: 28]  [ 2:103] 
	# PEDAL: [ 1: 36]  [ 1: 55] 
	
	if manual == 0:
		KEY_LEFT_NOTE = 48
		KEY_RIGHT_NOTE = 103
		LED_LEFT = LED_MANUAL*0.225
		LED_RIGHT = LED_MANUAL*0.99
		key_offset = note_no - KEY_LEFT_NOTE
		led_offset = LED_LEFT + (LED_RIGHT - LED_LEFT) / (KEY_RIGHT_NOTE - KEY_LEFT_NOTE + 1) * key_offset
		return clamp_led_index(int(led_offset))
	elif manual == 1:
		KEY_LEFT_NOTE = 28
		KEY_RIGHT_NOTE = 103
		LED_LEFT = LED_MANUAL*-0.045
		LED_RIGHT = LED_MANUAL*1.005
		key_offset = note_no - KEY_LEFT_NOTE
		led_offset = LED_LEFT + (LED_RIGHT - LED_LEFT) / (KEY_RIGHT_NOTE - KEY_LEFT_NOTE + 1) * key_offset
		return clamp_led_index(int(led_offset))
	elif manual == 2:
		pedal_map = [
			0, 1, 2, 3, 4,
			6, 7, 8, 9, 10, 11, 12
		]
		KEY_LEFT_NOTE = 36
		KEY_RIGHT_NOTE = 55
		LED_LEFT = LED_MANUAL*0.184
		LED_RIGHT = LED_MANUAL*0.932
		
		note_key = pedal_map[note_no % 12]
		note_oct = int(note_no / 12)
		note_loc = note_oct * 12 + note_key / 14 * 12
		
		key_offset = note_loc - KEY_LEFT_NOTE
		led_offset = LED_LEFT + (LED_RIGHT - LED_LEFT) / (KEY_RIGHT_NOTE - KEY_LEFT_NOTE + 1) * key_offset
		return clamp_led_index(int(led_offset))
	else:
		return 0


def clamp_led_index(n):
	return max(0, min(n, LED_MANUAL - 1))

def cleanup_note_event():
	now = time.perf_counter()
	while len(note_event) > 0:
		if note_event[0][4] < now - NOTE_EVENT_TTL:
			note_event.pop(0)
		else:
			break

def led_exp():
	
	# Exp 0 - 127 -> 0 - 30
	
	global led_array
	color = color_interp([0, 20, 40], [80, 0, 20], 0, 127, current_expression)
	fill_length = int(current_expression/127*22)
	led_idx = LED_MARGIN + int(LED_MANUAL/2) - fill_length
	led_array[3, LED_MARGIN : LED_MARGIN + fill_length] += color
	led_array[3, -LED_MARGIN  - fill_length : -LED_MARGIN] += color

def led_program_change():
	now = time.perf_counter()
	
	global led_array
	
	for event in note_event:
		(type, channel, note, sc, t, clk) = event
		if type == 'program_change':
			if clk >= midi_clock - 12:
				color = color_interp_period([10, 20, 20], [0, 0, 0], 0, 0.2, midi_clock - clk, 4)
				led_array[3, LED_MARGIN + 30: LED_MARGIN + LED_MANUAL - 30] += color


def led_note_on_off(note_on_color_shape, note_off_color_shape):
	now = time.perf_counter()
	
	for event in note_event:
		(type, channel, note, sc, t, clk) = event
		manual = channel_map[channel]
		if type == 'note_on':
			if clk >= midi_clock - 12:
				br = 1 - (midi_clock - clk) / 12 # 1 - 0
				if manual == 2: # pedal
					move = 2 + int(midi_clock - clk) # 0 - 12
					idx = note % 12
					draw_led_array_brightness(manual, note_to_index(manual, note) + move, note_color[idx:idx+1], br)
					draw_led_array_brightness(manual, note_to_index(manual, note) - move, note_color[idx:idx+1], br)
					if clk >= midi_clock - 6: # pedal
						draw_led_array_bg(manual, note_color[idx:idx+1], (midi_clock - clk)/6 * 0.4)
				else:
					move = int(len(note_on_color_shape)/2)+1 + int(midi_clock - clk) # 0 - 12
					draw_led_array_brightness(manual, note_to_index(manual, note) + move, note_on_color_shape, br)
					draw_led_array_brightness(manual, note_to_index(manual, note) - move, note_on_color_shape, br)
		if type == 'note_off':
			if clk >= midi_clock - 12:
				br = 1 - (midi_clock - clk) / 12 # 1 - 0
				if manual == 2: # pedal
					move = 2 + int(midi_clock - clk) # 0 - 12
					idx = note % 12
					draw_led_array_brightness(manual, note_to_index(manual, note) + move, note_color[idx:idx+1], br)
					draw_led_array_brightness(manual, note_to_index(manual, note) - move, note_color[idx:idx+1], br)
				else:
					move = int(len(note_off_color_shape)/2)+1 + int(midi_clock - clk) # 0 - 12
					draw_led_array_brightness(manual, note_to_index(manual, note) + move, note_off_color_shape, br)
					draw_led_array_brightness(manual, note_to_index(manual, note) - move, note_off_color_shape, br)

# 色、Upperノート範囲、Lowerノート範囲
def led_note_on_state(c0, c1, u0, u1, l0, l1):
	for (channel, note) in note_state:
		manual = channel_map[channel]
		if manual == 2: # pedal
			idx = note % 12
			draw_led_array(manual, note_to_index(manual, note), np.repeat(note_color[idx:idx+1], 5, axis=0))
		elif manual == 0: # upper
			color = color_interp(c0, c1, u0, u1, note)
			draw_led_array(manual, note_to_index(manual, note), np.repeat([color], 3, axis=0))
			draw_led_array_brightness(manual, note_to_index(manual, note)+3, np.repeat([color], 3, axis=0), 0.5)
			draw_led_array_brightness(manual, note_to_index(manual, note)-3, np.repeat([color], 3, axis=0), 0.5)
		else: # lower
			color = color_interp(c0, c1, l0, l1, note)
			draw_led_array(manual, note_to_index(manual, note), np.repeat([color], 3, axis=0))
			draw_led_array_brightness(manual, note_to_index(manual, note)+3, np.repeat([color], 3, axis=0), 0.5)
			draw_led_array_brightness(manual, note_to_index(manual, note)-3, np.repeat([color], 3, axis=0), 0.5)

# 色配列
def led_note_on_state_loop(clist):
	for (channel, note) in note_state:
		manual = channel_map[channel]
		if manual == 2: # pedal
			idx = note % 12
			draw_led_array(manual, note_to_index(manual, note), np.repeat(note_color[idx:idx+1], 5, axis=0))
		else: # upper/lower
			colors = np.array(clist, dtype=np.uint32)
			idx = scene_note_count[channel] % len(colors)
			draw_led_array(manual, note_to_index(manual, note), np.repeat(colors[idx:idx+1], 3, axis=0))
			draw_led_array_bg(manual, colors[idx:idx+1], 0.2)

# 最後の1音を固定色で左右へ広げる。現在のシーンになった以降の音に反応する。
# manual: 0=upper, 1=lower, 2=pedal
def led_note_on_off_wave(target_manual, color, br):
	now = time.perf_counter()
	
	last_note_on_idx = 0
	last_note_on_time = 0
	last_note_off_idx = 0
	last_note_off_time = 0
	for event in note_event:
		(type, channel, note, sc, t, clk) = event
		manual = channel_map[channel]
		if manual == target_manual and t >= scene_start_time:
			if type == 'note_on':
				last_note_on_idx = note_to_index(manual, note)
				last_note_on_time = t
			if type == 'note_off':
				last_note_off_idx = note_to_index(manual, note)
				last_note_off_time = t
	
	global led_array
	if last_note_on_idx > 0:
		width = min(LED_MARGIN, 1 + int((now - last_note_on_time) / 0.01))
		led_idx = LED_MARGIN + last_note_on_idx
		led_array[target_manual, led_idx - width : led_idx + width] += (np.array(color).astype(np.float32) * br).astype(np.uint32)
	if last_note_off_idx > 0:
		width = min(LED_MARGIN, 1 + int((now - last_note_off_time) / 0.01))
		led_idx = LED_MARGIN + last_note_on_idx
		led_array[target_manual, led_idx - width : led_idx + width] = np.array([0,0,0]).astype(np.uint32)

def led_scene():
	now = time.perf_counter()
	
	led_exp()
	led_program_change()
	if current_mode == 0:
		led_note_on_off(led_note_particle_shape, led_note_particle_shape)
		led_note_on_state([255, 0, 0], [0, 0, 255], 48, 103, 28, 103)
	if current_mode == 1:
		bpm = 96
		beat_time = (music_start_clock % 24) / 24
		
		# Beat
		if current_scene > 0 and current_scene < 12:
			draw_led_array_brightness(0, 2 + ((music_start_clock//24)%beat_in_a_bar) * len(led_note_shape_white) + len(led_note_shape_white)//2,
				led_note_shape_white, 1 - (music_start_clock%24)/24)
		
		# 全範囲 Upper 48-103 Lower 28-103
		# Upper B3-E5
		# Lower B2-C5
		if current_scene == 0:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
		elif current_scene == 1:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
		elif current_scene == 2:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state_loop([
				[  0,   0, 255],
				[  0, 255,   0],
				[  0, 255, 255],
				[255, 255, 255],
			])
		elif current_scene == 3:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
		elif current_scene == 4:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
		elif current_scene == 5:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			draw_led_array_bg(0, [0, 0, 80], (1 - beat_time * 2 % 1) * 0.2)
			draw_led_array_bg(1, [0, 0, 80], (1 - beat_time * 2 % 1) * 0.2)
		elif current_scene == 6:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			color = color_interp([0, 0, 80], [80, 0, 0], 0, 60/bpm * 0.8, now - scene_start_time)
			draw_led_array_bg(0, color, 0.2)
			draw_led_array_bg(1, color, 0.2)
		elif current_scene == 7:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			draw_led_array_bg(0, [80, 0, 0], (1 - beat_time * 2 % 1) * 0.2)
			draw_led_array_bg(1, [80, 0, 0], (1 - beat_time * 2 % 1) * 0.2)
		elif current_scene == 8:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			draw_led_array_bg(0, [80, 20, 0], (1 - beat_time * 2 % 1) * 0.3)
			draw_led_array_bg(1, [80, 20, 0], (1 - beat_time * 2 % 1) * 0.3)
		elif current_scene == 9:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			color = color_interp([80, 20, 0], [0, 0, 80], 0, 60/bpm * 3.3, now - scene_start_time)
			draw_led_array_bg(0, color, 0.2)
			draw_led_array_bg(1, color, 0.2)
		elif current_scene == 10:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			color = color_interp([0, 0, 80], [0, 0, 0], 0, 60/bpm * 3.5, now - scene_start_time)
			draw_led_array_bg(0, color, 0.2)
			draw_led_array_bg(1, color, 0.2)
		elif current_scene == 11:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 0, 0], [0, 0, 255],
				note_name_to_no['B3'], note_name_to_no['E5'], note_name_to_no['B2'], note_name_to_no['C5'])
			draw_led_array_bg(0, [20, 20, 20], (1 - beat_time % 1) * 0.3)
			draw_led_array_bg(1, [20, 20, 20], (1 - beat_time % 1) * 0.3)
		elif current_scene == 12:
			led_note_on_off(led_note_particle_shape, led_note_particle_shape)
			led_note_on_state([255, 255, 255], [255, 255, 255], 48, 103, 28, 103)
			led_note_on_off_wave(0, [255, 128, 0], 0.1) # 0=upper
	
	

def led_loop(led_lock,term_lock):
	try:
		while True:
			start_time = time.perf_counter()
			
			with led_lock:
				init_led_array()
				
				# Scene
				led_scene()
				
				global led_array
				np.clip(led_array, 0, 255, led_array)
				
				cleanup_note_event()
			
			show_led()
			
			global perf_led_lasttime
			now = time.perf_counter()
			perf_led.append(now - start_time)
			if perf_led_lasttime > 0:
				perf_led_intr.append(now - perf_led_lasttime)
			perf_led_lasttime = now
			
			time.sleep(0.001)
	#       time.sleep(0.1)
	except Exception as e:
		logging.exception('led_loop error', exc_info=e)
	finally:
		ws.ws2811_fini(leds)
		ws.delete_ws2811_t(leds)


#### Terminal Thread

on_notes_lower = []
on_notes_pedal = []

def show_chord():
	global on_notes_lower
	global on_notes_pedal
	
	new_on_notes_pedal = []
	new_on_notes_lower = []
	for (ch, note) in note_state:
		if ch == 1:
			new_on_notes_pedal.append(note % 12)
		elif ch == 2:
			new_on_notes_lower.append(note % 12)
	
	if len(new_on_notes_pedal) > 0:
		on_notes_pedal = new_on_notes_pedal
	if len(new_on_notes_lower) > 0:
		on_notes_lower = new_on_notes_lower
	
	if len(new_on_notes_pedal) == 0 and len(new_on_notes_lower) == 0:
		on_notes_pedal = []
		on_notes_lower = []
	
	name = {
		0: 'C',
		1: 'C#',
		2: 'D',
		3: 'D#',
		4: 'E',
		5: 'F',
		6: 'F#',
		7: 'G',
		8: 'G#',
		9: 'A',
		10: 'A#',
		11: 'B',
	}
	note_list = list(map(name.get, sorted(list(set(on_notes_pedal + on_notes_lower)))))
	
	print("\033[19;1HCHORD: \033[0K", end='');
	
	if len(note_list) == 0:
		return
	
	for chord in pychord.find_chords_from_notes(note_list):
		print("{} ".format(chord), end='')
	
	print("\033[19;40H | ", end='')
	for note in note_list:
		print("{} ".format(note), end='')
	
	

def show_terminal(led_array):
	
	print("\033[1;1H", end='')
	
	for ch in range(MANUAL_COUNT):
		print("|", end='')
		for led in led_array[ch, LED_MARGIN:LED_MARGIN+int(LED_MANUAL/2)]:
			print("\033[48;2;{};{};{}m ".format(led[0], led[1], led[2]), end='')
		print("<\033[0m\n    >", end='')
		for led in led_array[ch, LED_MARGIN+int(LED_MANUAL/2):LED_MARGIN+LED_MANUAL]:
			print("\033[48;2;{};{};{}m ".format(led[0], led[1], led[2]), end='')
		print("|\n", end='')
	
	print("\033[0m", end='')
	
	note_event_ch = [[] for i in range(4)]
	for note in note_event:
		# (type, channel, note, scene, time)
		if note[1] < 4:
			note_event_ch[note[1]].append(note)
	
	print("\033[9;1H", end='')
	for ch in range(4):
		print("{:1}{:1}|\033[0K".format(ch, ch_to_name(ch)), end='')
		for note in reversed(note_event_ch[ch][-12:]):
			# (type, channel, note, scene, time)
			if note[0] == 'note_on':
				print("\033[48;2;{};{};{}m\033[38;2;{};{};{}m".format(127, 0, 0, 255, 255, 255), end='')
			if note[0] == 'note_off':
				print("\033[48;2;{};{};{}m\033[38;2;{};{};{}m".format(0, 0, 127, 255, 255, 255), end='')
			print("[{:3}]".format(note_to_name(note[2])), end='')
		print("\033[0m\n", end='')
	
	show_chord()
	
	# debug
#	print("\033[18;1H", end='')
#	print(led_array[0, LED_MARGIN+40:LED_MARGIN+70])
	
	print("\033[20;1H\033[0K", end='');
	print("Exp: {:3} ".format(current_expression), end='');
	print("MIDI Event: {:8} ".format(midi_event_count), end='')
	print("NOTE Event Queue: {:6}".format(len(note_event)), end='')
	
	print("\033[16;1H\033[0K", end='')
	print("MODE: {}".format(mode_name[current_mode]), end='')
	
	# 
	if music_start_time > 0:
		now = time.perf_counter()
		print("\033[17;1H\033[0K", end='')
		print("TIME:{:7.2f} BAR:{:3} BEAT:{:4.2f}  ".format(now - music_start_time,
			music_start_clock//24//beat_in_a_bar,
			(music_start_clock//24)%beat_in_a_bar), end='')
		print("SCENE[{:2}]:{:7.2f} ".format(current_scene, now - scene_start_time), end='')
		print("COUNT:", end='')
		for count in scene_note_count[0:4]:
			print(" {:3}".format(count), end='')
	else:
		print("\033[17;1H\033[0K", end='')
	
	if music_start_clock > 0:
		print("\033[18;1H\033[0K", end='')
		q = music_start_clock//24
		bar = q // 4
		q %= 4
		print("CLOCK:{:5}.{:3}.{:3}  ({:8})".format(bar, q, music_start_clock % 24, midi_clock), end='')
	


def term_loop(led_lock,term_lock):
	try:
		while True:
			start_time = time.perf_counter()
			
			led_array_copy = None
			with led_lock:
				led_array_copy = led_array.copy()
			with term_lock:
				show_terminal(led_array_copy)
			
			global perf_term_lasttime
			now = time.perf_counter()
			perf_term.append(now - start_time)
			if perf_term_lasttime > 0:
				perf_term_intr.append(now - perf_term_lasttime)
			perf_term_lasttime = now
			
			time.sleep(0.01)
	#		time.sleep(0.05)
	except Exception as e:
		logging.exception('term_loop error', exc_info=e)


#### MIDI Thread

note_history = {}

for ch in range(16):
	note_history['note_on', ch] = []
	note_history['note_off', ch] = []
	

def check_scene_change(msg, t):
	note_history[msg.type, msg.channel].append(msg.note)
	if len(note_history[msg.type, msg.channel]) > MAX_NOTE_HISTORY:
		note_history[msg.type, msg.channel].pop(0)
	
	global current_mode
	global current_scene
	global midi_clock
	global music_start_time
	global music_start_clock
	global music_stop
	global scene_start_time
	global scene_note_count
	global current_expression
	global midi_event_count
	global note_state
	global note_event
	
	# Exit
	if msg.type == 'note_off' and msg.note == 0:
		print("\033[16;1H\033[0K", end='');
		print("EXIT")
		time.sleep(1)
		os._exit(0)
	
	# Scene reset
	if msg.type == 'note_on' and msg.channel == 2 and msg.note == 28:
		
		# Exit
		if (2, 29) in note_state and (2, 30) in note_state:
			print("\033[16;1H\033[0K", end='');
			print("EXIT")
			time.sleep(1)
			os._exit(0)
		
		current_scene = 0
		midi_clock = 0
		music_start_time = 0
		music_start_clock = 0
		music_stop = 0
		scene_start_time = 0
		scene_note_count = [0] * 16
		
		current_expression = 0
		midi_event_count = 0
		note_state = {}
		note_event = []
		
		return
	
	# Scene debug
	if msg.type == 'note_off' and msg.note == 1:
		scene_start_time = t
		scene_note_count = [0] * 16
		if current_scene == 0:
			music_start_time = t
			music_start_clock = 0
		current_scene -= 1
		return
	if msg.type == 'note_off' and msg.note == 2:
		scene_start_time = t
		scene_note_count = [0] * 16
		if current_scene == 0:
			music_start_time = t
			music_start_clock = 0
		current_scene += 1
		return
	
	# Mode Change
	if msg.type == 'note_on' and msg.channel == 2 and msg.note == 29:
		if (2, 28) in note_state:
			current_mode -= 1
			if current_mode < 0:
				current_mode = max(mode_name.keys())
			return
	if msg.type == 'note_on' and msg.channel == 2 and msg.note == 30:
		if (2, 28) in note_state:
			current_mode += 1
			if current_mode not in mode_name:
				current_mode = 0
			return
	
	if current_mode in scene_trigger:
		if (current_scene, msg.type, msg.channel) in scene_trigger[current_mode]:
			for (trigger_notes, next_scene) in scene_trigger[current_mode][current_scene, msg.type, msg.channel]:
				trigger_notes_no = list(map(note_name_to_no.get, trigger_notes))
				if note_history[msg.type, msg.channel][-len(trigger_notes):] == trigger_notes_no:
					scene_start_time = t
					scene_note_count = [0] * 16
					if current_scene == 0:
						music_start_time = t
						music_start_clock = 0
					current_scene = next_scene
	
	if msg.type == 'note_on':
		scene_note_count[msg.channel] += 1

def midi_loop(led_lock,term_lock, inport):
	try:
		for msg in inport:
			start_time = time.perf_counter()
		#	logger.debug("midi:{}".format(msg))
			with led_lock:
				global midi_event_count
				global music_stop
				midi_event_count += 1
				if msg.type == 'clock' and music_stop == 0:
					global midi_clock
					midi_clock += 1
					global music_start_clock
					music_start_clock += 1
				if msg.type == 'start':
					music_stop = 0
					music_start_clock = 0
				if msg.type == 'stop':
					music_stop = 1
				if msg.type == 'continue':
					music_stop = 0
				if msg.type == 'note_on':
					t = time.perf_counter()
					check_scene_change(msg, t)
					note_state[(msg.channel, msg.note)] = (current_scene, t)
					note_event.append(('note_on', msg.channel, msg.note, current_scene, t, midi_clock))
				#	print("on  ", msg.channel, msg.note, time.perf_counter())
				if msg.type == 'note_off':
					t = time.perf_counter()
					if (msg.channel, msg.note) in note_state:
						del note_state[(msg.channel, msg.note)]
					note_event.append(('note_off', msg.channel, msg.note, current_scene, t, midi_clock))
					check_scene_change(msg, t)
				#	print("off ", msg.channel, msg.note)
				if msg.is_cc(11): # Expression
					global current_expression
					current_expression = msg.value
				if msg.type == 'program_change':
					note_event.append(('program_change', msg.channel, 0, current_scene, time.perf_counter(), midi_clock))
			
			global perf_midi_lasttime
			now = time.perf_counter()
			perf_midi.append(now - start_time)
			if perf_midi_lasttime > 0:
				perf_midi_intr.append(now - perf_midi_lasttime)
			perf_midi_lasttime = now
	except Exception as e:
		logging.exception('term_loop error', exc_info=e)


#### Main Thread

led_lock = threading.Lock()
term_lock = threading.Lock()

led_thread = threading.Thread(target=led_loop, args=(led_lock,term_lock))
led_thread.deamon = True
led_thread.start()

inport = mido.open_input(MIDI_PORT)
midi_thread = threading.Thread(target=midi_loop, args=(led_lock,term_lock,inport))
midi_thread.deamon = True
midi_thread.start()

term_thread = threading.Thread(target=term_loop, args=(led_lock,term_lock))
term_thread.deamon = True
term_thread.start()


print("\033[2J\033[?25l", end='');
while True:
	with term_lock:
		print("\033[13;1H", end='');
		led = np.asarray(perf_led)
		led_intr = np.asarray(perf_led_intr)
		if len(led) and len(led_intr):
			print("LED : {:3} fps (intr: {:3.0f}, max: {:3.0f}, mean: {:3.0f})".format(
				len(led), np.max(led_intr)*1000, np.max(led)*1000, np.mean(led)*1000))
		
		print("\033[14;1H", end='');
		midi = np.asarray(perf_midi)
		midi_intr = np.asarray(perf_midi_intr)
		if len(midi) and len(midi_intr):
			print("MIDI: {:3} fps (intr: {:3}, max: {:3.0f}, mean: {:3.0f})".format(
				len(midi), "---", np.max(midi)*1000, np.mean(midi)*1000))
		
		print("\033[15;1H", end='');
		term = np.asarray(perf_term)
		term_intr = np.asarray(perf_term_intr)
		if len(term) and len(term_intr):
			print("TERM: {:3} fps (intr: {:3.0f}, max: {:3.0f}, mean: {:3.0f})".format(
				len(term), np.max(term_intr)*1000, np.max(term)*1000, np.mean(term)*1000))
		
		perf_led = []
		perf_midi = []
		perf_term = []
	
	time.sleep(1)




