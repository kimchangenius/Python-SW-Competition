import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch

import config as cfg
from env import SumoEnv
from mappo_agent import MAPPOAgent

# 기본 경로 설정 (config 활용)
SITE_ROOT = cfg.SITE_ROOT
SUMO_ROOT = cfg.SUMO_ROOT
DATA_ROOT = cfg.DATA_ROOT

# SUMO 실행 파일/도구 경로
sumo_bin_path = SUMO_ROOT / "bin"
sumo_tools_path = SUMO_ROOT / "tools"
is_windows = os.name == "nt"
sumo_path = sumo_bin_path / ("sumo.exe" if is_windows else "sumo")
sumo_gui_path = sumo_bin_path / ("sumo-gui.exe" if is_windows else "sumo-gui")
netconvert_path = sumo_bin_path / ("netconvert.exe" if is_windows else "netconvert")

# 환경 변수 및 traci import 준비
os.environ.setdefault("SUMO_HOME", str(SUMO_ROOT))
sys.path.append(str(sumo_tools_path))
import traci

# 시뮬레이션에 사용할 입력 파일들 (config 활용)
NET_FILE = cfg.NET_FILE
NETCONFIG_FILE = DATA_ROOT / "rect8.netccfg"
ROUTE_FILE = cfg.ROUTE_FILE
TLS_FILE = cfg.TLS_FILE


def ensure_network_file() -> None:
    """netconvert로 네트워크 파일을 매번 재생성."""
    if not NETCONFIG_FILE.exists():
        raise FileNotFoundError(f"네트워크 설정 파일을 찾을 수 없습니다: {NETCONFIG_FILE}")
    if not netconvert_path.exists():
        raise FileNotFoundError(f"netconvert 바이너리를 찾을 수 없습니다: {netconvert_path}")

    if NET_FILE.exists():
        NET_FILE.unlink()
        print(f"기존 네트워크 파일을 삭제합니다: {NET_FILE}")

    result = subprocess.run(
        [
            str(netconvert_path),
            "-c",
            str(NETCONFIG_FILE),
            "--output-file",
            str(NET_FILE),
        ],
        cwd=str(DATA_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"netconvert 실패:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    print(f"네트워크 파일 생성 완료: {NET_FILE}")


def build_sumo_command(use_gui: bool = False) -> List[str]:
    """SUMO 실행 커맨드 리스트를 구성."""
    exe = sumo_gui_path if use_gui else sumo_path
    if not exe.exists():
        raise FileNotFoundError(f"SUMO 실행 파일을 찾을 수 없습니다: {exe}")

    for required in [NET_FILE, ROUTE_FILE, TLS_FILE]:
        if not required.exists():
            raise FileNotFoundError(f"필수 입력 파일이 없습니다: {required}")

    cmd = [
        str(exe),
        "-n",
        str(NET_FILE),
        "-r",
        str(ROUTE_FILE),
        "-a",
        str(TLS_FILE),
        "--step-length",
        "1.0",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    # GUI 실행 시 시각적 속도 조절을 위해 delay(밀리초)를 설정
    if use_gui:
        cmd += ["--delay", "100"]
    return cmd


def apply_tls_program(program_id: str = "real") -> None:
    """추가 파일에 정의된 신호 프로그램을 모든 TLS에 적용."""
    tls_ids = list(traci.trafficlight.getIDList())
    missing: List[str] = []

    for tls_id in tls_ids:
        available = [logic.programID for logic in traci.trafficlight.getAllProgramLogics(tls_id)]
        if program_id in available:
            traci.trafficlight.setProgram(tls_id, program_id)
        else:
            # 원하는 프로그램이 없으면 기존 기본 프로그램을 유지하고 경고만 남김
            missing.append(f"{tls_id} (programs: {', '.join(available)})")

    if missing:
        print(f"경고: 지정한 신호 프로그램 '{program_id}'을 찾을 수 없습니다 -> {', '.join(missing)}")


def write_result_csv(name: str, mode: str, steps: int, total_delay: float, avg_queue: dict) -> None:
    """시뮬레이션 결과를 result/sim_run_results.csv에 누적 저장."""
    result_dir = cfg.SITE_ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_file = result_dir / f"{name}.csv"
    write_header = not out_file.exists()
    import csv

    with open(out_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["mode", "steps", "total_delay", "queue_json"])
        writer.writerow([mode, steps, f"{total_delay:.2f}", avg_queue])


def run_simulation(max_steps: Optional[int] = None, use_gui: bool = True) -> None:
    """Traci를 사용해 시뮬레이션을 실행."""
    ensure_network_file()
    cmd = build_sumo_command(use_gui=use_gui)
    print("SUMO 실행:", " ".join(cmd))
    traci.start(cmd)
    apply_tls_program(program_id="real")

    step = 0
    delay_accum = 0.0
    step_length = float(traci.simulation.getDeltaT())
    queue_sum = {}
    queue_steps = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            try:
                wait_now = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
                delay_accum += wait_now * step_length

                for lane in traci.lane.getIDList():
                    q = traci.lane.getLastStepHaltingNumber(lane)
                    queue_sum[lane] = queue_sum.get(lane, 0.0) + q
                queue_steps += 1
            except traci.TraCIException:
                pass
            traci.simulationStep()
            step += 1
            if max_steps and step >= max_steps:
                print(f"max_steps({max_steps}) 도달, 시뮬레이션을 종료합니다.")
                break
        avg_queue = {lane: (total / queue_steps if queue_steps else 0.0) for lane, total in queue_sum.items()}
        print(f"시뮬레이션 종료 - 총 step: {step}, 총 delay(초·lane합): {delay_accum:.2f}")
        write_result_csv("static_simulation", "static", step, delay_accum, avg_queue)
    finally:
        traci.close(False)


def run_simulation_with_model(
    model_path: Optional[Path] = None,
    max_steps: Optional[int] = None,
    use_gui: bool = True,
) -> None:
    if model_path is None:
        model_path = cfg.SITE_ROOT / "models" / "mappo_latest.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # SUMO 네트워크 준비
    ensure_network_file()

    # 환경 및 에이전트 초기화
    env = SumoEnv(use_gui=use_gui, step_length=cfg.DEFAULT_STEP_LENGTH)
    obs = env.reset()

    conf = cfg.MAPPOConfig()
    obs_dim = obs[cfg.TLS_IDS[0]].shape[0]
    action_dim = len(conf.action_phases)
    agent = MAPPOAgent(agent_ids=cfg.TLS_IDS, obs_dim=obs_dim, action_dim=action_dim, config=conf)

    # 체크포인트 로드
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        if "actor" in state:
            agent.actor.load_state_dict(state["actor"])
        if "critic" in state:
            agent.critic.load_state_dict(state["critic"])
    print(f"모델 로드 완료: {model_path}")

    step = 0
    try:
        done = False
        while not done:
            actions, _, _ = agent.select_actions(obs)
            obs, rewards, done = env.step(actions)
            step += 1

            if max_steps and step >= max_steps:
                print(f"max_steps({max_steps}) 도달, 시뮬레이션을 종료합니다.")
                break
        print(f"모델 기반 시뮬레이션 종료 - 총 step: {step}, 총 delay(초·lane합): {env.delay_accum:.2f}")
        write_result_csv("mappo_simulation", "model", step, env.delay_accum, env.get_avg_queue_lengths())
    finally:
        env.close()


if __name__ == "__main__":
    print("traci 경로:", Path(traci.__file__).resolve())
    run_simulation()
    # run_simulation_with_model()
