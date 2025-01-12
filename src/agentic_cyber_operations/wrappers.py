import logging
import inspect
import math
from copy import deepcopy
from typing import Union, List
from prettytable import PrettyTable
import numpy as np
from gymnasium import spaces, Env

from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import TrueTableWrapper
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Shared import Results
from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent
from agents import RedAgent, BlueAgent
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4

# this wrapper converts a list into an action object based on the action space
from CybORG.Shared.Actions import Sleep


class MultiAgentIntListToActionWrapper(BaseWrapper):
    def __init__(self, env=None, agent=None):
        super().__init__(env, agent)
        self.action_space = None
        self.action_params = None
        self.param_name = None
        self.selection_mask = None

    def step(self, agent=None, action: list = None) -> Results:
        if action is not None:
            action_obj = self.get_action(agent, action)
        else:
            action_obj = None
        result = self.env.step(agent, action_obj)
        result.action_space, result.selection_masks = self.action_space_change(
            result.action_space
        )
        self.selection_mask = result.selection_masks
        result.action_name = str(action_obj)
        return result

    def reset(self, agent=None, **kwargs):
        result = self.env.reset(agent, **kwargs)
        result.action_space, result.selection_masks = self.action_space_change(
            result.action_space
        )
        self.selection_mask = result.selection_masks
        result.observation = self.observation_change(observation=result.observation)
        return result

    def get_action_space(self, agent: str) -> dict:
        action_space, selection_mask = self.action_space_change(
            self.env.get_action_space(agent)
        )
        self.selection_mask = selection_mask
        return action_space

    def action_space_change(self, action_space: dict) -> (list, list):
        self.action_space = action_space
        selection_masks = []
        new_action_space = []
        self.param_name = []
        for key, value in action_space.items():
            if len(value) > 1:
                new_action_space.append(len(value))
                selection_masks.append(
                    [list(value.keys()).index(i) for i, v in value.items() if v]
                )
                self.param_name.append(key)
        return new_action_space, selection_masks

    def get_action(self, agent: str, action: list):
        """converts a list to an action object"""
        if not isinstance(action, list):
            raise TypeError("action must be a list")
        opts = {}
        self.action_space = self.get_action_space(agent)
        action_class = list(self.action_space["action"])[action[0]]
        if self.action_params is None:
            self.action_params = {}
            for ac in self.action_space["action"].keys():
                self.action_params[ac] = inspect.signature(ac).parameters
        count = 0
        for key, value in self.action_space.items():
            if key in self.action_params[action_class]:
                if len(value) > 1:
                    if action[count] < len(value):
                        if list(value.values())[action[count]]:
                            opts[key] = list(value.keys())[action[count]]
                else:
                    if list(value.values())[0]:
                        opts[key] = list(value.keys())[0]
            if len(value) > 1:
                count += 1
        try:
            action_obj = action_class(**opts)
        except TypeError:
            action_obj = Sleep()

        # Reset for future use
        self.action_space = None
        self.action_params = None
        return action_obj

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)


class MultiAgentGymWrapper(Env, BaseWrapper):
    """
    OpenAI Gym wrapper for running multiple agents simultaneously.
    """

    def __init__(self, env: BaseWrapper = None):
        super().__init__(env)
        # Instantiate Red Agent
        if isinstance(self.get_action_space("Red"), list):
            red_action_space = spaces.MultiDiscrete(self.get_action_space("Red"))
            red_action_size = len(red_action_space)
        else:
            red_action_space = spaces.Discrete(self.get_action_space("Red"))
            red_action_size = red_action_space.n
        red_box_len = len(
            self.observation_change(observation=self.env.reset("Red").observation)
        )
        red_observation_space = spaces.Box(
            -1.0, 1.0, shape=(red_box_len,), dtype=np.float32
        )
        self.red_agent = RedAgent(action_size=red_action_size, state_size=red_box_len)
        # Instantiate Blue Agent
        if isinstance(self.get_action_space("Blue"), list):
            blue_action_space = spaces.MultiDiscrete(self.get_action_space("Blue"))
            blue_action_size = len(blue_action_space)
        else:
            blue_action_space = spaces.Discrete(self.get_action_space("Blue"))
            blue_action_size = blue_action_space.n
        blue_box_len = len(
            self.observation_change(observation=self.env.reset("Blue").observation)
        )
        blue_observation_space = spaces.Box(
            -1.0, 1.0, shape=(blue_box_len,), dtype=np.float32
        )
        self.blue_agent = BlueAgent(
            action_size=blue_action_size, state_size=blue_box_len
        )
        # Instantiate Green Agent
        self.green_agent = GreenAgent()
        self.reward_range = (float("-inf"), float("inf"))
        self.metadata = dict()
        self.agents = {
            "Red": self.red_agent,
            "Blue": self.blue_agent,
            "Green": self.green_agent,
        }
        self.observation_spaces = {
            "Red": red_observation_space,
            "Blue": blue_observation_space,
        }
        self.action_spaces = {"Red": red_action_space, "Blue": blue_action_space}
        self.uuid = str(uuid4())
        self.writer = SummaryWriter(log_dir=f"./logs/{self.uuid}")
        self.action_space = None
        self.observation_space = None
        self.action = None

    def step(self, agent: str = None, action: Union[int, List[int]] = None):
        self.action_space = self.action_spaces[agent]
        self.observation_space = self.observation_spaces[agent]
        self.action = action
        result = self.env.step(agent, action)
        result.observation = self.observation_change(observation=result.observation)
        result.action_space = self.action_space_change(result.action_space)
        terminated = result.done
        truncated = False
        info = vars(result)
        return np.array(result.observation), result.reward, terminated, truncated, info

    def reset(self, agent=None, **kwargs):
        result = self.env.reset(agent=agent, **kwargs)
        result.action_space = self.action_space_change(result.action_space)
        result.observation = self.observation_change(result.observation)
        return np.array(result.observation), vars(result)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent: str):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent: str):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()


class MultiAgentTableWrapper(BaseWrapper):
    """
    Table wrapper for running multiple agents simultaneously.
    """

    def __init__(self, env=None, output_mode="table"):
        super().__init__(env)
        red_env = TrueTableWrapper(env=env, agent="Red")
        blue_env = TrueTableWrapper(env=env, agent="Blue")
        self.envs = {"Red": red_env, "Blue": blue_env}
        self.active_agent = None
        self.red_info = dict()
        self.blue_info = dict()
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.output_mode = output_mode
        self.success = None
        self.baseline = None
        self.action_record = {"Red": [], "Blue": []}

    def reset(self, agent=None, **kwargs):
        self.active_agent = agent
        self.env = self.envs[agent]
        result = self.env.reset(agent, **kwargs)
        if agent == "Red":
            self.red_info = {}
            self.known_subnets = set()
            self.step_counter = -1
            self.id_tracker = -1
            self.success = None
            result.observation = self.observation_change(observation=result.observation)
        elif agent == "Blue":
            self._process_initial_obs(agent="Blue", observation=result.observation)
            result.observation = self.observation_change(observation=result.observation)
        return result

    def get_table(self, agent=None):
        if agent == "Red":
            return self._create_red_table()
        elif agent == "Blue":
            return self._create_blue_table(success=None)
        elif agent is None:
            return self.env.get_table()

    def observation_change(self, observation, baseline=False):
        if self.active_agent == "Red":
            self.success = observation["success"]
            self.step_counter += 1
            if self.step_counter <= 0:
                self._process_initial_obs(agent="Red", observation=observation)
            elif self.success:
                self._update_red_info(observation)

            if self.output_mode == "table":
                obs = self._create_red_table()
            elif self.output_mode == "vector":
                obs = self._create_vector(agent="Red")
            elif self.output_mode == "raw":
                obs = observation
            else:
                raise NotImplementedError("Invalid output_mode")
            return obs
        if self.active_agent == "Blue":
            obs = deepcopy(observation)
            success = obs["success"]
            self._process_last_action()
            anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
            del obs["success"]
            info = self._process_anomalies(anomaly_obs)
            if baseline:
                for host in info:
                    info[host][-2] = "None"
                    info[host][-1] = "No"
                    self.blue_info[host][-1] = "No"

            self.blue_info = info

            if self.output_mode == "table":
                return self._create_blue_table(success)
            elif self.output_mode == "anomaly":
                anomaly_obs["success"] = success
                return anomaly_obs
            elif self.output_mode == "raw":
                return observation
            elif self.output_mode == "vector":
                return self._create_vector(agent="Blue", success=success)
            else:
                raise NotImplementedError("Invalid output_mode for BlueTableWrapper")

    def _process_initial_obs(self, agent, observation):
        if agent == "Red":
            for hostid in observation:
                if hostid == "success":
                    continue
                host = observation[hostid]
                interface = host["Interface"][0]
                subnet = interface["Subnet"]
                self.known_subnets.add(subnet)
                ip = str(interface["IP Address"])
                hostname = host["System info"]["Hostname"]
                self.red_info[ip] = [
                    str(subnet),
                    str(ip),
                    hostname,
                    False,
                    "Privileged",
                ]

        if agent == "Blue":
            observation = observation.copy()
            self.baseline = observation
            del self.baseline["success"]
            for hostid in observation:
                if hostid == "success":
                    continue
                host = observation[hostid]
                interface = host["Interface"][0]
                subnet = interface["Subnet"]
                ip = str(interface["IP Address"])
                hostname = host["System info"]["Hostname"]
                self.blue_info[hostname] = [
                    str(subnet),
                    str(ip),
                    hostname,
                    "None",
                    "No",
                ]
            return self.blue_info

    def _process_last_action(self):
        action = self.get_last_action(agent="Blue")
        if action is not None:
            name = action.__class__.__name__
            self.action_record["Blue"].append(name)
            hostname = (
                action.get_params()["hostname"]
                if name in ("Restore", "Remove")
                else None
            )

            if name == "Restore":
                self.blue_info[hostname][-1] = "No"
            elif name == "Remove":
                compromised = self.blue_info[hostname][-1]
                if compromised != "No":
                    self.blue_info[hostname][-1] = "Unknown"

    def _detect_anomalies(self, obs):
        if self.baseline is None:
            raise TypeError(
                "BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method."
            )

        anomaly_dict = {}

        for hostid, host in obs.items():
            if hostid == "success":
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if "Files" in host:
                baseline_files = host_baseline.get("Files", [])
                anomalous_files = []
                for f in host["Files"]:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies["Files"] = anomalous_files

            if "Processes" in host:
                baseline_processes = host_baseline.get("Processes", [])
                anomalous_processes = []
                for p in host["Processes"]:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies["Processes"] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict

    def _process_anomalies(self, anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if "Processes" in host_anomalies:
                if "Connections" in host_anomalies["Processes"][-1]:
                    connection_type = self._interpret_connections(
                        host_anomalies["Processes"]
                    )
                    info[hostid][-2] = connection_type
                    if connection_type == "Exploit":
                        info[hostid][-1] = "User"
                        self.blue_info[hostid][-1] = "User"
            if "Files" in host_anomalies:
                malware = [f["Density"] >= 0.9 for f in host_anomalies["Files"]]
                if any(malware):
                    info[hostid][-1] = "Privileged"
                    self.blue_info[hostid][-1] = "Privileged"

        return info

    def _interpret_connections(self, activity: list):
        num_connections = len(activity)

        ports = set(
            [
                item["Connections"][0]["local_port"]
                for item in activity
                if "Connections" in item
            ]
        )
        port_focus = len(ports)

        remote_ports = set(
            [
                item["Connections"][0].get("remote_port")
                for item in activity
                if "Connections" in item
            ]
        )
        if None in remote_ports:
            remote_ports.remove(None)

        ######################################################################
        # NOTE: fixed bug relating to incorrectly classifying exploit as scan
        ######################################################################

        if 4444 in remote_ports:
            anomaly = "Exploit"
        elif num_connections >= 3 and port_focus >= 3:
            anomaly = "Scan"
        elif num_connections >= 3 and port_focus == 1:
            anomaly = "Exploit"
        elif "Service Name" in activity[0]:
            anomaly = "None"
        else:
            anomaly = "Scan"

        return anomaly

    def _update_red_info(self, obs):
        original_obs = deepcopy(obs)
        action = self.get_last_action(agent="Red")
        name = action.__class__.__name__
        self.action_record["Red"].append(name)
        if name == "DiscoverRemoteSystems":
            self._add_ips(obs)
        elif name == "DiscoverNetworkServices":
            try:
                ip = str(obs.popitem()[1]["Interface"][0]["IP Address"])
            except TypeError as e:
                obs["Success"] = False
                logging.warning(
                    f"Encountered a TypeError when trying to get IP address for DiscoverRemoteSystems action."
                )
                return
            try:
                self.red_info[ip][3] = True
            except Exception as e:
                logging.warning(
                    f"Encountered an error when trying to update red_info for DiscoverNetwork Services on {ip}. "
                    f"Original observation: {original_obs} "
                    f"red_info: {self.red_info} "
                    f"Error: {e}"
                )
                obs["Success"] = False
        elif name == "ExploitRemoteService":
            self._process_exploit(obs)
        elif name == "PrivilegeEscalate":
            hostname = action.hostname
            self._process_priv_esc(obs, hostname)

    def _generate_name(self, datatype: str):
        self.id_tracker += 1
        unique_id = "UNKNOWN_" + datatype + ": " + str(self.id_tracker)
        return unique_id

    def _add_ips(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            for interface in host["Interface"]:
                ip = interface["IP Address"]
                subnet = interface["Subnet"]
                if subnet not in self.known_subnets:
                    self.known_subnets.add(subnet)
                if str(ip) not in self.red_info:
                    subnet = self._get_subnet(ip)
                    hostname = self._generate_name("HOST")
                    self.red_info[str(ip)] = [subnet, str(ip), hostname, False, "None"]
                elif self.red_info[str(ip)][0].startswith("UNKNOWN_"):
                    self.red_info[str(ip)][0] = self._get_subnet(ip)

    def _get_subnet(self, ip):
        for subnet in self.known_subnets:
            if ip in subnet:
                return str(subnet)
        return self._generate_name("SUBNET")

    def _process_exploit(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue

            host = obs[hostid]
            if "Sessions" in host:
                ip = str(host["Interface"][0]["IP Address"])
                if ip not in self.red_info.keys():
                    logging.warning(
                        f"Attempted to process exploit for unknown IP {ip}."
                    )
                    return
                hostname = host["System info"]["Hostname"]
                session = host["Sessions"][0]
                access = "Privileged" if "Username" in session else "User"

                self.red_info[ip][2] = hostname
                self.red_info[ip][4] = access

    def _process_priv_esc(self, obs, hostname):
        if obs["success"] == False:
            try:
                for ip_address, info in self.red_info.items():
                    if info[2] == hostname:
                        info[4] = "None"
            except IndexError as e:
                logging.warning(
                    f"IndexError in _process_priv_esc when trying to process observation failure. "
                    f"obs: {obs} "
                    f"hostname: {hostname} "
                    f"red_info: {self.red_info}"
                )
                return
        else:
            try:
                for hostid in obs:
                    if hostid == "success":
                        continue
                    host = obs[hostid]
                    ip = host["Interface"][0]["IP Address"]

                    if "Sessions" in host:
                        try:
                            access = "Privileged"
                            self.red_info[str(ip)][4] = access
                        except KeyError as e:
                            # Sometimes the game observes success when we do not have access to the system!
                            logging.warning(
                                f"KeyError in _process_priv_esc when trying to update access. {ip} not in red_info."
                            )
                            # Sometimes we get a "success" when it should not be successful!
                            subnet = str(obs[hostid]["Interface"][0]["Subnet"])
                            ip = str(ip)
                            hostname = hostid
                            scanned = False
                            access = "None"
                            self.red_info[str(ip)] = [
                                subnet,
                                ip,
                                hostname,
                                scanned,
                                access,
                            ]

                    else:
                        subnet = self._get_subnet(ip)
                        hostname = self._generate_name("HOST")

                        if str(ip) not in self.red_info:
                            self.red_info[str(ip)] = [
                                subnet,
                                str(ip),
                                hostname,
                                False,
                                "None",
                            ]
                        else:
                            self.red_info[str(ip)][0] = subnet
                            self.red_info[str(ip)][2] = hostname
            except IndexError as e:
                if "ip" not in locals():
                    ip = "an unspecified ip"
                logging.warning(
                    f"IndexError in _process_priv_esc when trying to assign access to {ip}."
                    f"hostname: {hostname}"
                    f"obs: {obs} "
                    f"red_info:{self.red_info}"
                )
            except KeyError as e:
                if "ip" not in locals():
                    ip = "an unspecified ip"
                logging.warning(
                    f"KeyError in _process_priv_esc when trying to assign access to {ip}."
                    f"hostname: {hostname} "
                    f"obs: {obs} "
                    f"red_info:{self.red_info}"
                )

    def _create_red_table(self):
        # The table data is all stored inside the ip nodes
        # which form the rows of the table
        table = PrettyTable(
            [
                "Subnet",
                "IP Address",
                "Hostname",
                "Scanned",
                "Access",
            ]
        )
        for ip in self.red_info:
            table.add_row(self.red_info[ip])

        table.sortby = "IP Address"
        table.success = self.success
        return table

    def _create_blue_table(self, success):
        table = PrettyTable(
            ["Subnet", "IP Address", "Hostname", "Activity", "Compromised"]
        )
        for hostid in self.blue_info:
            table.add_row(self.blue_info[hostid])

        table.sortby = "Hostname"
        table.success = success
        return table

    def _create_vector(self, agent, num_hosts=13, success=True):
        if agent == "Red":
            table = self._create_red_table()._rows

            # Compute required length of vector based on number of hosts
            padding = num_hosts - len(table)
            id_length = math.ceil(math.log2(num_hosts))

            success_value = int(self.success.value) if self.success.value < 2 else -1
            proto_vector = [success_value]
            for row in table:
                # Scanned
                proto_vector.append(int(row[3]))

                # Access
                access = row[4]
                if access == "None":
                    value = [0, 0]
                elif access == "User":
                    value = [1, 0]
                elif access == "Privileged":
                    value = [0, 1]
                else:
                    raise ValueError("Table had invalid Access Level")
                proto_vector.extend(value)

            proto_vector.extend(padding * 3 * [-1])

            return np.array(proto_vector)
        elif agent == "Blue":
            table = self._create_blue_table(success)._rows

            proto_vector = []
            for row in table:
                # Activity
                activity = row[3]
                if activity == "None":
                    value = [0, 0]
                elif activity == "Scan":
                    value = [1, 0]
                elif activity == "Exploit":
                    value = [1, 1]
                else:
                    raise ValueError("Table had invalid Access Level")
                proto_vector.extend(value)

                # Compromised
                compromised = row[4]
                if compromised == "No":
                    value = [0, 0]
                elif compromised == "Unknown":
                    value = [1, 0]
                elif compromised == "User":
                    value = [0, 1]
                elif compromised == "Privileged":
                    value = [1, 1]
                else:
                    raise ValueError("Table had invalid Access Level")
                proto_vector.extend(value)

            return np.array(proto_vector)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        self.active_agent = agent
        if self.output_mode == "raw":
            obs = self.get_attr("get_observation")(agent)
        elif self.output_mode == "table":
            obs = self.get_table(agent=agent)
        elif self.output_mode == "vector":
            obs = self._create_vector(agent=agent)
        else:
            raise NotImplementedError("Invalid output_mode")

        return obs

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent):
        return self.get_attr("get_action_space")(agent)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()


class MultiAgentChallengeWrapper(Env, BaseWrapper):
    def __init__(self, env, reward_threshold=None, max_steps=None, initial_agent="Red"):
        super().__init__(env)
        env = MultiAgentTableWrapper(env, output_mode="vector")
        env = EnumActionWrapper(env)
        env = MultiAgentGymWrapper(env)

        self.env = env
        self.action_spaces = self.env.action_spaces
        self.observation_spaces = self.env.observation_spaces
        self.agents = self.env.agents
        self.writer = self.env.writer
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = 0
        self.action_space = self.action_spaces[initial_agent]
        self.observation_space = self.observation_spaces[initial_agent]

    def step(self, agent, action):
        obs, reward, terminated, truncated, info = self.env.step(agent, action)

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter > self.max_steps:
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, agent, **kwargs):
        self.step_counter = 0
        return self.env.reset(agent, **kwargs)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent: str):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent: str):
        return self.env.get_last_action(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr("get_reward_breakdown")(agent)
