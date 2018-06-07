#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:58:03 2018

@author: ageorge
"""
from bob.pad.base.database import PadDatabase, PadFile
from bob.bio.video import FrameSelector
from bob.extension import rc

from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

import json

import os

import bob.io.base

import numpy as np


class BatlAggregatedPadFile(PadFile):
	"""
	A high level implementation of the File class for the BATL Aggregated
	database.
	"""

	def __init__(self, f,
				 stream_type,  # a list of streams to be loaded
				 max_frames,
				 reference_stream_type="color",
				 warp_to_reference=True,
				 convert_to_rgb=False,
				 crop=None,
				 video_data_only=True):

		"""
		**Parameters:**

		``f`` : :py:class:`object`
			An instance of the File class defined in the low level db interface
			of the BATL database, in the ``bob.db.batl.models.py`` file.

		``stream_type`` : [] or :py:class:`str`
			A types of the streams to be loaded.

		``max_frames`` : :py:class:`int`
			A maximum number of frames to be loaded. Frames are
			selected uniformly.

		``reference_stream_type`` : :py:class:`str`
			Align/register all channels to this one.
			Default: "color".

		``warp_to_reference`` : :py:class:`bool`
			Align/register downloaded video to ``reference_stream_type``,
			if set to ``True``.
			Default: ``True``.

		``convert_to_rgb`` : :py:class:`bool`
			Type cast the non-RGB data to RGB data type,
			if set to ``True``.
			Default: ``False``.

		``crop`` : []
			Pre-crop the frames if given, see ``bob.db.batl`` for more
			details.
			Default: ``None``.

		``video_data_only`` : :py:class:`bool`
			Load video data only if ``True``. Otherwise more meta-data
			is loaded, for example timestamps for each frame.
			See the ``load()`` method in the low-level database
			interface for more details.
			Default: ``True``.
		"""

		self.f = f
		if f.is_attack():
			attack_type = 'attack'
		else:
			attack_type = None

		file_id = self.encode_file_id(f)

		super(BatlAggregatedPadFile, self).__init__(
			client_id=f.client_id,
			path=f.path,
			attack_type=attack_type,
			file_id=file_id)

		self.stream_type = stream_type
		self.max_frames = max_frames
		self.reference_stream_type = reference_stream_type  # "color"
		self.warp_to_reference = warp_to_reference  # True
		self.convert_to_rgb = convert_to_rgb  # False
		self.crop = crop  # None
		self.video_data_only = video_data_only  # True

	def load(self, directory=None, extension='.h5',
			 frame_selector=FrameSelector(selection_style='all')):
		"""
		Load method of the file class.

		**Parameters:**

		``directory`` : :py:class:`str`
			String containing the path to BATL database.
			Default: ``None``.

		``extension`` : :py:class:`str`
			Extension of the BATL database.
			Default: ".h5".

		``frame_selector`` : :any:`bob.bio.video.FrameSelector`, optional
			Specifying the frames to be selected.

		**Returns:**

		``data`` : FrameContainer
			Video data stored in the FrameContainer,
			see ``bob.bio.video.utils.FrameContainer``
			for further details.
		"""

		# directory is a tuple

		if isinstance(self.f, bob.db.batl.models.VideoFile):

			directory_sel=directory.split(" ")[0]

		if isinstance(self.f, bob.db.batlgovt.models.VideoFile): 

			directory_sel=directory.split(" ")[1]

		#print("directory_sel",directory_sel)

		data = self.f.load(directory=directory_sel,
						   extension=extension,
						   modality=self.stream_type,
						   reference_stream_type=self.reference_stream_type,
						   warp_to_reference=self.warp_to_reference,
						   convert_to_rgb=self.convert_to_rgb,
						   crop=self.crop,
						   max_frames=self.max_frames)

		for meta_data in data.keys():
			if meta_data != 'rppg':
				data[meta_data] = frame_selector(data[meta_data])

		if self.video_data_only:

			data = data['video']

		return data

	def encode_file_id(self, f, n=3000):
		"""
		Return a modified version of the ``f.id`` ensuring uniqueness of the ids
		across all databases.

		**Parameters:**

		``f`` : :py:class:`object`
			An instance of the File class defined in the low level db interface


		``n`` : :py:class:`int`
			An offset to be added to the file id for different databases is defined
			as follows: offset = k*n, where k is the database number,
			k = 0,1,2 in our case. Default: 3000.

		**Returns:**

		``file_id`` : :py:class:`int`
			A modified version of the file id, which is now unigue accross
			all databases.
		"""

		import bob.db.batl
		import bob.db.batlgovt

		if isinstance(f, bob.db.batl.models.VideoFile): # check if instance of File class of LLDI of Replay-Attack
				
			file_id = f.id

		if isinstance(f, bob.db.batlgovt.models.VideoFile): 
			file_id = np.int(f.id + n)
			

		return file_id

class BatlAggregatedPadDatabase(PadDatabase):
	"""
	A high level implementation of the Database class for the BATL Aggregated
	database.
	"""

	def __init__(
			self,
			protocol='nowig__baseline', # accepts 2 for the time being
			original_directory=rc['bob.db.batl.directory'],
			original_extension='.h5',
			annotations_temp_dir="",
			landmark_detect_method="mtcnn",
			exlude_attacks_list=None,
			**kwargs):
		"""
		**Parameters:**

		``protocol`` : str or None
			The name of the protocol that defines the default experimental
			setup for this database. Also a "complex" protocols can be
			parsed.
			For example:
			"nowig-color-5" - nowig protocol, color data only,
			use 5 first frames.
			"nowig-depth-5" - nowig protocol, depth data only,
			use 5 first frames.
			"nowig-color" - nowig protocol, depth data only, use all frames.
			"nowig-infrared-50-join_train_dev" - nowig protocol,
			infrared data only, use 50 frames, join train and dev sets forming
			a single large training set.
			See the ``parse_protocol`` method of this class.

		``original_directory`` : str
			The directory where the original data of the database are stored.

		``original_extension`` : str
			The file name extension of the original data.

		``annotations_temp_dir`` : str
			Annotations computed in ``self.annotations(f)`` method of this
			class will be save to this directory if path is specified /
			non-empty string.
			Default: ``""``.

		``landmark_detect_method`` : str
			Method to be used to compute annotations - face bounding box and
			landmarks. Possible options: "dlib" or "mtcnn".
			Default: ``"mtcnn"``.

		``exlude_attacks_list`` : [str]
			A list of strings defining which attacks should be excluded from
			the training set. This shoould be handled in ``objects()`` method.
			Currently handled attacks: "makeup".
			Default: ``None``.

		``kwargs`` : dict
			The arguments of the :py:class:`bob.bio.base.database.BioDatabase`
			base class constructor.
		"""

		import bob.db.batl

		import bob.db.batlgovt


		self.batl_db=bob.db.batl.Database()

		self.batl_govt_db=bob.db.batlgovt.Database()

		# Since the high level API expects different group names than what the
		# low level API offers, you need to convert them when necessary
		self.low_level_group_names = (
			'train', 'validation',
			'test')  # group names in the low-level database interface
		self.high_level_group_names = (
			'train', 'dev',
			'eval')  # names are expected to be like that in objects() function

		# Always use super to call parent class methods.
		super(BatlAggregatedPadDatabase, self).__init__(
			name='batl_aggregated_db',
			protocol=protocol,
			original_directory=original_directory,
			original_extension=original_extension,
			**kwargs)

		#print("protocolsINIT:",protocol[0],protocol[0])
		self.protocol = protocol #tuple
		self.original_directory = original_directory #tuple
		self.original_extension = original_extension
		self.annotations_temp_dir = annotations_temp_dir #tuple
		self.landmark_detect_method = landmark_detect_method
		self.exlude_attacks_list = exlude_attacks_list

	# @property
	# def original_directory(self):
	# 	return self.batl_db.original_directory, self.batl_govt_db.original_directory

	# @original_directory.setter
	# def original_directory(self, value):
	# 	self.batl_db.original_directory = value.split(" ")[0]
	# 	self.batl_govt_db.original_directory = value.split(" ")[0]

	def parse_protocol(self, protocol):
		# Handle a single protocol statement here

		"""
		Parse the protocol name, which is give as a string.
		An example of protocols it can parse:
		"nowig-color-5" - nowig protocol, color data only, use 5 first frames.
		"nowig-depth-5" - nowig protocol, depth data only, use 5 first frames.
		"nowig-color" - nowig protocol, depth data only, use all frames.

		**Parameters:**

		``protocol`` : str
			Protocol name to be parsed. Example: "nowig-depth-5" .

		**Returns:**

		``protocol`` : str
			The name of the protocol as defined in the low level db interface.

		``stream_type`` : str
			The name of the channel/stream_type to be loaded.

		``max_frames`` : int
			The number of frames to be loaded.

		``extra`` : str
			An extra string which is handled in ``self.objects()`` method.
			Extra strings which are currently handled are defined in
			``possible_extras`` of this function.
			For example, if ``extra="join_train_dev"``, the train and dev
			sets will be joined in ``self.objects()``,
			forming a single training set.
		"""

		possible_extras = ['join_train_dev','trainon_idiap_teston_gov','trainon_gov_teston_idiap','trainon_both_teston_gov','trainon_both_teston_idiap','trainon_idiap_teston_idiap','trainon_gov_teston_gov','trainon_both_teston_gov_realgov','trainon_both_teston_idiap_realgov']


		# Here exclusing make up is generally used

		#print("protocol:",protocol)

		components = protocol.split("-")

		#print("components:",components)

		extra = [item for item in possible_extras if item in components]

		extra = extra[0] if extra else None

		if extra is not None:
			components.remove(extra)

		components += [None, None]

		components = components[0:3]

		protocol, stream_type, max_frames = components

		if max_frames is not None:

			max_frames = int(max_frames)

		return protocol, stream_type, max_frames, extra


	def objects(self,
				protocol=None,
				groups=None,
				purposes=None,
				model_ids=None,
				**kwargs):
		"""
		This function returns lists of BatlPadFile objects, which fulfill the
		given restrictions.

		**Parameters:**

		``protocol`` : str
			The protocol for which the clients should be retrieved.
			The protocol is dependent on your database.
			If you do not have protocols defined, just ignore this field.

		``groups`` : :py:class:`str`
			OR a list of strings.
			The groups of which the clients should be returned.
			Usually, groups are one or more elements of ('train', 'dev', 'eval')

		``purposes`` : :obj:`str` or [:obj:`str`]
			The purposes for which File objects should be retrieved.
			Usually it is either 'real' or 'attack'.

		``model_ids``
			This parameter is not supported in PAD databases yet

		**Returns:**

		``files`` : [BatlPadFile]
			A list of BATLPadFile objects.
		"""

		

		if protocol is None:
			protocol = self.protocol

		if groups is None:
			groups = self.high_level_group_names

		if purposes is None:
			purposes = ['real', 'attack']

		#print("protocols:",protocol[0],protocol[1])

		protocol_batl, stream_type, max_frames, extra_batl = self.parse_protocol(protocol.split("__")[0])
		protocol_gvt_batl, stream_type, max_frames, extra_gvt_batl = self.parse_protocol(protocol.split("__")[1])

		# Convert group names to low-level group names here.
		groups = self.convert_names_to_lowlevel(
			groups, self.low_level_group_names, self.high_level_group_names)

		if not isinstance(groups, list) and groups is not None:  # if a single group is given make it a list
			groups = list(groups)

		
		###########

		#print("extra_batl:",extra_batl,"extra_gvt_batl:",extra_gvt_batl)

		batl_govt_files=[]

		batl_files=[]

		####### outer as a base


		if extra_batl is None:
			batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=groups,
										purposes=purposes, **kwargs)

			batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
											groups=groups,
											purposes=purposes, **kwargs)




		if extra_batl is not None and "join_train_dev" in extra_batl:
			#print('INSIDE',groups)

			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs) 

			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['test'],
										purposes=purposes, **kwargs)

	


		########
		if extra_gvt_batl is not None and "join_train_dev" in extra_gvt_batl:
			#print('INSIDE',groups)

			if groups == ['train']: # join "train" and "dev" sets
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)

			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['test'],
										purposes=purposes, **kwargs)





		############## Handling protocols assumes that both have same protocol extra so taking batl one, later implement something instead of tuple to one string as scores go to the stuff


		######### PROTOCOL 1##################################################


		if extra_gvt_batl is not None and "trainon_idiap_teston_gov" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)


			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['test'],
										purposes=purposes, **kwargs)



		######### PROTOCOL2 ################################################## 

		if extra_gvt_batl is not None and "trainon_gov_teston_idiap" in extra_gvt_batl:

			## Train on a db means combining train and validation




			if groups == ['train']: # join "train" and "dev" sets
				#print("1.train:groups",groups)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				#print("2.train:groups",groups)


				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_files = self.batl_db.objects(protocol=protocol_batl, ## added now AG
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)


			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				#print("3.test:groups",groups)
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['test'],
										purposes=purposes, **kwargs)



		######### PROTOCOL3 ##################################################'','','trainon_both_teston_idiap'

		if extra_gvt_batl is not None and "trainon_both_teston_gov" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['test'],
										purposes=purposes, **kwargs)




		######### PROTOCOL4 ################################################## '','',''

		if extra_gvt_batl is not None and "trainon_both_teston_idiap" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['test'],
										purposes=purposes, **kwargs)




		######### PROTOCOL5 ################################################## '','',''

		if extra_gvt_batl is not None and "trainon_idiap_teston_idiap" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)


			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['test'],
										purposes=purposes, **kwargs)



		######### PROTOCOL6 ################################################## '','',''

		if extra_gvt_batl is not None and "trainon_gov_teston_gov" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)


			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['test'],
										purposes=purposes, **kwargs)


		

		######### PROTOCOL7 ##################################################'','','trainon_both_teston_idiap'

		if extra_gvt_batl is not None and "trainon_both_teston_gov_realgov" in extra_gvt_batl:

			# Train with both but exclude all attacks from training in gov test data

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				
				## use only bonafides from govt for training

				batl_govt_files=[ f for f in batl_govt_files if f.type_id == 0]




			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['test'],
										purposes=purposes, **kwargs)
		######### PROTOCOL8 ##################################################'','','trainon_both_teston_idiap'


		if extra_gvt_batl is not None and "trainon_both_teston_idiap_realgov" in extra_gvt_batl:

			## Train on a db means combining train and validation


			if groups == ['train']: # join "train" and "dev" sets
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=['train', 'validation'],
										purposes=purposes, **kwargs)
				## use only bonafides from govt for training
				
				batl_govt_files=[ f for f in batl_govt_files if f.type_id == 0]


			# return ALL data if "train" and "some other" set/sets are requested
			elif len(groups)>=2 and 'train' in groups:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

				batl_govt_files = self.batl_govt_db.objects(protocol=protocol_gvt_batl,
										groups=self.low_level_group_names,
										purposes=purposes, **kwargs)

			# addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
			else:
				batl_files = self.batl_db.objects(protocol=protocol_batl,
										groups=['test'],
										purposes=purposes, **kwargs)







		################## PROTOCOL ENDS #################################





		sfiles=batl_govt_files+batl_files

		print("LenBATL:",len(batl_files),"LenGOV:",len(batl_govt_files))


		## Exclude makeup only from training set, while it is requested for training

		if groups == 'train' or 'train' in groups and len(groups) == 1:
			# exclude "makeup" case
			if self.exlude_attacks_list is not None and "makeup" in self.exlude_attacks_list:


				tfiles=[]

				for f in sfiles:


					if isinstance(f, bob.db.batl.models.VideoFile):

						if os.path.split(f.path)[-1].split("_")[-2:-1][0] != "5":

							tfiles.append(f)

					if isinstance(f, bob.db.batlgovt.models.VideoFile): 

						if f.pai_id != 4:

							tfiles.append(f)
				sfiles=tfiles

		files = [BatlAggregatedPadFile(f, stream_type, max_frames) for f in sfiles]

		return files

	def annotations(self, f):

		print("annot",f)
		"""
		Computes annotations for a given file object ``f``, which
		is an instance of the ``BatlPadFile`` class.

		NOTE: you can pre-compute annotation in your first experiment
		and then reuse them in other experiments setting
		``self.annotations_temp_dir`` path of this class, where
		precomputed annotations will be saved.

		**Parameters:**

		``f`` : :py:class:`object`
			An instance of ``BatlPadFile`` defined above.

		**Returns:**

		``annotations`` : :py:class:`dict`
			A dictionary containing annotations for
			each frame in the video.
			Dictionary structure:
			``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
			Where
			``frameN_dict`` contains coordinates of the
			face bounding box and landmarks in frame N.
		"""

		import bob.db.batl

		import bob.db.batlgovt

		annotations = {}

		#print("type",type(f.f)) 

		if isinstance(f.f, bob.db.batl.models.VideoFile):

			#print("Idiap instance")

			file_path = os.path.join(self.annotations_temp_dir.split(" ")[0], f.f.path + ".json")

			if not os.path.isfile(file_path):  # no file with annotations

				f.stream_type = "color"
				f.reference_stream_type = "color"
				f.warp_to_reference = False
				f.convert_to_rgb = False
				f.crop = None
				f.video_data_only = True

				video = f.load(directory=self.original_directory.split(" ")[0],
							   extension=self.original_extension)

				annotations = {}

				for idx, image in enumerate(video.as_array()):

					frame_annotations = detect_face_landmarks_in_image(image, method=self.landmark_detect_method)

					if frame_annotations:

						annotations[str(idx)] = frame_annotations

				if self.annotations_temp_dir.split(" ")[0]:  # if directory is not an empty string

					bob.io.base.create_directories_safe(directory=os.path.split(file_path)[0], dryrun=False)

					with open(file_path, 'w+') as json_file:

						json_file.write(json.dumps(annotations))

			else:  # if file with annotations exists load them from file

				with open(file_path, 'r') as json_file:

					annotations = json.load(json_file)

	

		if isinstance(f.f, bob.db.batlgovt.models.VideoFile): 
			#print("Gov instance")
			#print("self.annotations_temp_dir",self.annotations_temp_dir)
	  

			file_path = os.path.join(self.annotations_temp_dir.split(" ")[1], f.f.path + ".json")

			if not os.path.isfile(file_path):  # no file with annotations
				#print("No annotations found!!")

				f.stream_type = "color"
				f.reference_stream_type = "color"
				f.warp_to_reference = False
				f.convert_to_rgb = False
				f.crop = None
				f.video_data_only = True

				video = f.load(directory=self.original_directory.split(" ")[1],
							   extension=self.original_extension)

				annotations = {}



				for idx, image in enumerate(video.as_array()):

					frame_annotations = detect_face_landmarks_in_image(image, method=self.landmark_detect_method)

					if frame_annotations:

						annotations[str(idx)] = frame_annotations

				if self.annotations_temp_dir.split(" ")[1]:  # if directory is not an empty string

					bob.io.base.create_directories_safe(directory=os.path.split(file_path)[0], dryrun=False)

					with open(file_path, 'w+') as json_file:

						json_file.write(json.dumps(annotations))

			else:  # if file with annotations exists load them from file
				#print("Annotations found!!")

				with open(file_path, 'r') as json_file:

					annotations = json.load(json_file)


		## Annotation return

		if not annotations:  # if dictionary is empty

			return None
		#else:
		return annotations


		

