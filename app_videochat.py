import logging
import math
from typing import List
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import binascii
import io
from binascii import unhexlify
from aiortc.contrib.media import MediaPlayer, MediaRecorder

try:
	from typing import Literal
except ImportError:
	from typing_extensions import Literal  # type: ignore

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_server_state import server_state, server_state_lock

from streamlit_webrtc import (
	ClientSettings,
	MixerBase,
	VideoProcessorBase,
	WebRtcMode,
	WebRtcStreamerContext,
	create_mix_track,
	create_process_track,
	webrtc_streamer,
	AudioProcessorBase,
    AudioProcessorFactory,
	VideoProcessorFactory,
	MediaRecorderFactory,
	MediaPlayerFactory,
	AudioHTMLAttributes,
)

logger = logging.getLogger(__name__)


KEY_SIZE = 16
KEY = b'3' * KEY_SIZE
IV = b'1' * KEY_SIZE


def pad32(data):
	return pad(data, block_size=KEY_SIZE)
	# n_bytes = len(data)
	# no = int(np.ceil(n_bytes / KEY_SIZE))
	# n32 = no * KEY_SIZE - n_bytes
	# return data + b'0' * n32


class EncryptVideo(VideoProcessorBase):
	type: Literal["noop", "cartoon", "edges", "rotate"]

	def __init__(self) -> None:
		self.type = "noop"
		super(EncryptVideo, self).__init__()
		self.cipher = AES.new(key=KEY, mode=AES.MODE_CBC, iv=IV)

	def recv(self, frame: av.VideoFrame):

		img = frame.to_ndarray(format="bgr24")

		if self.type == "noop":
			pass
		elif self.type == "cartoon":
			# prepare color
			img_color = cv2.pyrDown(cv2.pyrDown(img))
			for _ in range(6):
				img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
			img_color = cv2.pyrUp(cv2.pyrUp(img_color))

			# prepare edges
			img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img_edges = cv2.adaptiveThreshold(
				cv2.medianBlur(img_edges, 7),
				255,
				cv2.ADAPTIVE_THRESH_MEAN_C,
				cv2.THRESH_BINARY,
				9,
				2,
			)
			img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

			# combine color and edges
			img = cv2.bitwise_and(img_color, img_edges)
		elif self.type == "edges":
			# perform edge detection
			img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
		elif self.type == "rotate":
			# rotate image
			rows, cols, _ = img.shape
			M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
			img = cv2.warpAffine(img, M, (cols, rows))

		# l = img.shape[0] * img.shape[1] * img.shape[2]
		# buf = bytearray(img)
		# buf = pad32(buf)
		# buf = self.cipher.encrypt(buf)
		# buf = np.frombuffer(buf[:l], dtype='uint8')
		# buf = buf.reshape(img.shape)
		frame = av.VideoFrame.from_ndarray(img, format="bgr24")
		return frame

	async def recv_queued(self, frames: List[av.VideoFrame]):
		"""
		Receives all the frames arrived after the previous recv_queued() call
		and returns new frames when running in async mode.
		If not implemented, delegated to the recv() method by default.
		"""
		# if len(frames) > 1:
		#     logger.warning(
		#         "Some frames have been dropped during audio processing. "
		#         "`recv_queued` is recommended to use instead."
		#     )
		return [self.recv(frames[-1])]


class EncryptAudio(AudioProcessorBase):

	def __init__(self) -> None:
		super(EncryptAudio, self).__init__()
		self.cipher = AES.new(key=KEY, mode=AES.MODE_CBC, iv=IV)

		self.resampler = av.audio.resampler.AudioResampler(format="s16", layout='mono', rate=22050)

	def recv(self, frame: av.AudioFrame):
		# pts = frame.pts
		#
		# try:
		# 	frame.pts = None
		# 	frame = self.resampler.resample(frame)
		# except:
		# 	logger.warning('Resampling was not performed')
		# frame.pts = pts
		rate = frame.rate
		img = frame.to_ndarray()
		data = bytearray(img)
		data = pad32(data[::4])  # [:len(data) - len(BEG_AUDIO)])
		data_dec = self.cipher.encrypt(data)

		buf = np.frombuffer(data_dec, dtype='int8')
		buf.dtype = 'int16'
		frm = frame.from_ndarray(buf.reshape((1, -1)))
		frm.rate = rate // 4
		return frm

	async def recv_queued(self, frames: List[av.AudioFrame]):
		"""
		Receives all the frames arrived after the previous recv_queued() call
		and returns new frames when running in async mode.
		If not implemented, delegated to the recv() method by default.
		"""
		return [self.recv(frames[-1])]


class MultiWindowMixer(MixerBase):

	def __init__(self) -> None:
		self.key = KEY
		self.cipher = AES.new(key=KEY, mode=AES.MODE_CBC, iv=IV)

	def on_update(self, frames):
		buf_w = 640
		buf_h = 480
		buffer = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

		n_inputs = len(frames)

		n_cols = math.ceil(math.sqrt(n_inputs))
		if n_cols == 0:
			return None

		n_rows = math.ceil(n_inputs / n_cols)
		grid_w = buf_w // n_cols
		grid_h = buf_h // n_rows

		for i in range(n_inputs):
			frame = frames[i]
			if frame is None:
				continue

			grid_x = (i % n_cols) * grid_w
			grid_y = (i // n_cols) * grid_h

			img = frame.to_ndarray(format="bgr24")
			src_h, src_w = img.shape[0:2]

			aspect_ratio = src_w / src_h

			window_w = min(grid_w, int(grid_h * aspect_ratio))
			window_h = min(grid_h, int(window_w / aspect_ratio))

			window_offset_x = (grid_w - window_w) // 2
			window_offset_y = (grid_h - window_h) // 2

			window_x0 = grid_x + window_offset_x
			window_y0 = grid_y + window_offset_y
			window_x1 = window_x0 + window_w
			window_y1 = window_y0 + window_h
			# l = img.shape[0] * img.shape[1] * img.shape[2]
			# buf = bytearray(img)
			# # buf = unpad(buf, block_size=KEY_SIZE)
			# buf = self.cipher.decrypt(buf)
			#
			# buf = np.frombuffer(buf[:l], dtype='uint8')
			#
			# try:
			# 	img = buf.reshape(img.shape)
			# except:
			# 	continue
			buffer[window_y0:window_y1, window_x0:window_x1, :] = cv2.resize(
				img, (window_w, window_h)
			)

		new_frame = av.VideoFrame.from_ndarray(buffer, format="bgr24")

		return new_frame


class MultiAudioMixer(MixerBase):

	def __init__(self) -> None:
		super(MultiAudioMixer, self).__init__()
		self.cipher = AES.new(key=KEY, mode=AES.MODE_CBC, iv=IV)
		_empty = np.zeros((1, 1920), dtype='int16')
		self.empty_frame = av.AudioFrame.from_ndarray(_empty, layout='stereo', format='s16')
		self.empty_frame.rate = 48000
		self.empty_frame.pts = 0

	def on_update(self, frames):

		# if frames[0].samples > 2000:
		# 	return frames[0]
		frames = sorted(frames, key=lambda x: x.pts)
		frame_all = None
		pts = 0
		for ind, frame in enumerate(frames[-1:]):
			rate = frame.rate
			buf = frame.to_ndarray()
			data = bytearray(buf)
			try:
				data_dec = self.cipher.decrypt(data)
				buf = np.frombuffer(data_dec, dtype='int8')
				buf.dtype = 'int16'
				# self.empty_frame = frm
				# if frame_all is None:
				frame_all = buf
				pts = frame.pts

				frm = av.AudioFrame.from_ndarray(frame_all.reshape((1, -1)), format='s16', layout='stereo')
				frm.rate = rate
				frm.pts = pts
				return frm

				# else:
				# 	frame_all = np.mean(np.append(frame_all.reshape((1, -1)), buf.reshape((1, -1)), axis=1), axis=0)
				# 	frame_all = np.int16(frame_all)  # np.append(frame_all, buf, axis=0)
			except:
				continue

		# if frame_all is not None:
		# 	frm = av.AudioFrame.from_ndarray(frame_all.reshape((1, -1)), format='s16', layout='stereo')
		# 	frm.rate = rate
		# 	frm.pts = pts
		# 	return frm

		return self.empty_frame


def main():
	with server_state_lock["webrtc_contexts"]:
		if "webrtc_contexts" not in server_state:
			server_state["webrtc_contexts"] = []

	with server_state_lock["mix_track"]:
		if "mix_track" not in server_state:
			server_state["mix_track"] = create_mix_track(
				kind="video", mixer_factory=MultiWindowMixer, key="mix"
			)
	#
	# with server_state_lock["mix_audio"]:
	# 	if "mix_audio" not in server_state:
	# 		server_state["mix_audio"] = create_mix_track(
	# 			kind="audio", mixer_factory=MultiAudioMixer, key="mixAudio"
	# 		)

	# with server_state_lock["mix_audio_send"]:
	# 	if "mix_audio_send" not in server_state:
	# 		server_state["mix_audio_send"] = create_mix_track(
	# 			kind="audio", mixer_factory=MultiAudioMixerSend, key="mixAudioSend"
	# 		)

	mix_track = server_state["mix_track"]
	# mix_audio = server_state["mix_audio"]
	# mix_audio_send = server_state["mix_audio_send"]

	self_ctx_video = webrtc_streamer(
		key="self",
		mode=WebRtcMode.SENDRECV,
		client_settings=ClientSettings(
			rtc_configuration={
				"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			},
			media_stream_constraints={"video": True, "audio": True},
		),
		source_video_track=mix_track,
		# source_audio_track=mix_audio,
		# audio_html_attrs=AudioHTMLAttributes(advanced={'sampleRate': 16000}),
		# in_recorder_factory=MediaRecorder,
		# desired_playing_state=True,
		sendback_audio=False,
	)

	self_process_video = None
	if self_ctx_video.input_video_track:
		self_process_video = create_process_track(
			input_track=self_ctx_video.input_video_track,
			processor_factory=EncryptVideo,
		)
		mix_track.add_input_track(self_process_video)
		self_process_video.processor.type = st.radio(
			"Select transform type",
			("noop", "cartoon", "edges", "rotate"),
			key="filter1-type",
		)
	#
	# self_process_audio = None
	# if self_ctx_video.input_audio_track:
	# 	self_process_audio = create_process_track(
	# 		input_track=self_ctx_video.input_audio_track,
	# 		processor_factory=EncryptAudio,
	# 	)
	# 	mix_audio.add_input_track(self_process_audio)

	with server_state_lock["webrtc_contexts"]:
		webrtc_contexts: List[WebRtcStreamerContext] = server_state["webrtc_contexts"]
		self_is_playing = self_ctx_video.state.playing  and self_process_video
		if self_is_playing and self_ctx_video not in webrtc_contexts:
			webrtc_contexts.append(self_ctx_video)
			server_state["webrtc_contexts"] = webrtc_contexts
		elif not self_is_playing and self_ctx_video in webrtc_contexts:
			webrtc_contexts.remove(self_ctx_video)
			server_state["webrtc_contexts"] = webrtc_contexts

		# for ctx in webrtc_contexts:
		# 	if ctx == self_ctx_video:
		# 		continue
		# 	webrtc_contexts.remove(ctx)

	# Audio streams are transferred in SFU manner
	# TODO: Create MCU to mix audio streams
	for ctx in webrtc_contexts:
		if ctx == self_ctx_video or ctx.state.playing is False:
			continue
		webrtc_streamer(
			key=f"sound-{id(ctx)}",
			mode=WebRtcMode.RECVONLY,
			client_settings=ClientSettings(
				rtc_configuration={
					"iceServers": [{"urls": ["stun:stun2.l.google.com:19302"]}]
				},
				media_stream_constraints={"video": False, "audio": True},
			),
			source_audio_track=ctx.input_audio_track,
			desired_playing_state=ctx.state.playing,
			# sendback_audio=False
		)


if __name__ == "__main__":
	import os

	DEBUG = os.environ.get("INFO", "false").lower() not in ["false", "no", "0"]

	logging.basicConfig(
		format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
		"%(message)s",
		force=True,
	)

	logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

	st_webrtc_logger = logging.getLogger("streamlit_webrtc")
	st_webrtc_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

	aioice_logger = logging.getLogger("aioice")
	aioice_logger.setLevel(logging.WARNING)

	fsevents_logger = logging.getLogger("fsevents")
	fsevents_logger.setLevel(logging.WARNING)

	main()
