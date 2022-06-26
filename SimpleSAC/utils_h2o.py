import random
import pprint
import time
import uuid
import tempfile
import os
import re
from copy import copy
from socket import gethostname
import pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb

import torch


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = ''
        config.project = 'when-to-trust-your-simulator'
        config.entity = 't6-thu'
        config.output_dir = './experiment_output'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)
        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )

# update user flags with flags_def
def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output



def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

# generate xml assets path: gym_xml_path
def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ",xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./xml_path/real_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)

#TODO: gravity
def update_target_env_gravity(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_gravityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../xml_path/real_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../xml_path/sim_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../xml_path/sim_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

#TODO: density
def update_target_env_density(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_densityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../xml_path/real_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../xml_path/sim_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../xml_path/sim_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

#TODO: friction
def update_target_env_friction(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_frictionx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../xml_path/real_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../xml_path/sim_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../xml_path/sim_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)


# def generate_log(extra=None):
#     print(extra)
#     record_data('../documents/log_{}.txt'.format(args.log_index), "{}".format(extra))