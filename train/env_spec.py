from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from train.utils import default_empty_dict, generate_scaled_trace


@dataclass
class ConfigLink:
    # Path to the trace file. If relative, will be considered to be relative to the
    # `traces_dir` folder.
    # Note that it is ok *not* to set a trace file for the donwlink: in that case
    # it will re-use the uplink trace file with the same scaling factor (regardless
    # of the value of the downlink's trace scaling factor).
    trace: str = ""
    # Absolute path to the trace file.
    trace_abs_path: str = "${resolve_path: ${.trace}, ${traces_dir}}"
    # Scaling factor for the trace: higher means sending packets faster (e.g., a
    # scaling factor of 2 means that packets are sent twice as fast as in the trace).
    trace_scaling_factor: float = 1.0
    # See definition in `pantheon_env.py`.
    min_scaled_trace_size: int = "${min_scaled_trace_size}"
    # Path to the scaled trace file. It will automatically filled based on the
    # `trace` and its `trace_scaling_factor`, when generating the scaled trace file.
    scaled_trace: str = ""
    # Loss passed to the `mm-loss {up,down}link` command. Ignored if <0.
    loss: float = -1
    # Type of queue (`--{up,down}link-queue=T`). Ignored if empty.
    queue_type: str = ""
    # Size of the queue in packets (`--{up,down}link-queue-args=packets=N`).
    # Ignored if <0.
    queue_size_packets: int = -1
    # Size of the queue in bytes (`--{up,down}link-queue-args=bytes=N`).
    # Ignored if <0.
    queue_size_bytes: int = -1


@dataclass
class ConfigEnvSpec:
    # Optional description of the environment (for documentation purpose).
    desc: str = ""
    # The `job_id` is the index in the list of jobs. It will be filled at runtime
    # when the config is loaded.
    job_id: Optional[int] = None
    # Delay passed to the `mm-delay` command. It is added to both up/downlinks.
    # Ignored if <0.
    delay: int = -1
    # Standard deviation of the noise added to RTT measurements.
    rtt_noise_std: float = 0.0
    # Uplink config.
    uplink: ConfigLink = ConfigLink()
    # Downlink config.
    downlink: ConfigLink = ConfigLink()


@dataclass
class ConfigJobs:
    """Config holding multiple environment specifications, defining a set of jobs"""

    jobs: Dict[str, ConfigEnvSpec] = default_empty_dict()


def get_env_cmd(flags, env: ConfigEnvSpec) -> List[str]:
    """
    Return the command line to launch the desired Pantheon scenario.
    """
    # Common arguments for all environments.
    cmd = [flags.test_path] + list(flags.common_params)

    prepend_mm_cmds = []
    extra_mm_link_args = []

    if env.delay >= 0:
        prepend_mm_cmds += ["mm-delay", str(env.delay)]

    for link_name in ["uplink", "downlink"]:
        link_cmd = get_link_cmd(link_name, env[link_name])
        prepend_mm_cmds += link_cmd.pop("--prepend-mm-cmds", [])
        extra_mm_link_args += link_cmd.pop("--extra-mm-link-args", [])
        # Other link arguments can be appended directly to the command line.
        cmd += [a for args in link_cmd.items() for a in args]

    if prepend_mm_cmds:
        cmd += ["--prepend-mm-cmds", " ".join(prepend_mm_cmds)]

    if extra_mm_link_args:
        cmd += ["--extra-mm-link-args", " ".join(extra_mm_link_args)]

    return cmd


def get_link_cmd(link_name: str, link: ConfigLink) -> Dict[str, List[str]]:
    """
    Obtain command line parameters for a given link.

    The returned dictionary maps a command line flag (e.g., "--prepend-mm-cmds") to
    its value (e.g., "mm-loss uplink 0.01").

    `link_name` should be either `uplink` or `downlink`.
    """
    # Obtain path to the trace file.
    cmd = {f"--{link_name}-trace": link.scaled_trace}

    if link.loss > 0:
        cmd["--prepend-mm-cmds"] = ["mm-loss", link_name, str(link.loss)]

    cmd["--extra-mm-link-args"] = extra_mm_link_args = []

    if link.queue_type:
        extra_mm_link_args.append(f"--{link_name}-queue={link.queue_type}")

    if link.queue_size_packets >= 0:
        extra_mm_link_args.append(
            f"--{link_name}-queue-args=packets={link.queue_size_packets}"
        )

    if link.queue_size_bytes >= 0:
        assert link.queue_size_packets < 0  # currently not supporting both
        extra_mm_link_args.append(
            f"--{link_name}-queue-args=bytes={link.queue_size_bytes}"
        )

    return cmd


def set_scaled_traces(job: ConfigEnvSpec, trace_dir: Path):
    """
    Update the input `job` with information on the scaled trace file.
    """
    for link_type in ["uplink", "downlink"]:  # careful, order matters!
        link = job[link_type]
        if link_type == "downlink" and (
            not link.trace  # means we want to use the same settings as the uplink
            or (
                link.trace_abs_path == job.uplink.trace_abs_path
                and link.trace_scaling_factor == job.uplink.trace_scaling_factor
            )
        ):
            # Downlink can re-use the same scaled trace as uplink.
            link.scaled_trace = job.uplink.scaled_trace

        elif link.trace_scaling_factor == 1:
            # Simple case where the trace is not actually scaled.
            link.scaled_trace = link.trace_abs_path

        else:
            # Create a rescaled trace file.
            base_trace = Path(link.trace_abs_path)
            scaled_trace = trace_dir / f"x{link.trace_scaling_factor}_{base_trace.name}"
            assert not scaled_trace.exists()
            generate_scaled_trace(
                base_trace,
                scaled_trace,
                link.trace_scaling_factor,
                link.min_scaled_trace_size,
            )
            link.scaled_trace = scaled_trace
