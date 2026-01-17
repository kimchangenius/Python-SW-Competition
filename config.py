import os
from pathlib import Path
from dataclasses import dataclass


# 경로 설정 (SUMO 루트/데이터 파일)
SITE_ROOT = Path(__file__).resolve().parent
SUMO_ROOT = SITE_ROOT / "sumo-1.19.0"
DATA_ROOT = SITE_ROOT / "data"

NET_FILE = DATA_ROOT / "rect8.net.xml"
ROUTE_FILE = DATA_ROOT / "rect8.rou.xml"
TLS_FILE = DATA_ROOT / "tls.add.xml"

# 제어할 신호 ID들 (네트워크의 tlsLogic id와 일치)
TLS_IDS = ["J1", "J2", "J3", "J4"]

# SUMO 실행 설정
DEFAULT_STEP_LENGTH = 1.0
USE_GUI_DEFAULT = False  # 학습 시에는 GUI 비활성화, 평가/시연 시만 GUI 사용


@dataclass
class MAPPOConfig:
    seed: int = 0
    total_episodes: int = 300
    rollout_steps: int = 200  # 한 에피소드 동안 수집할 max step
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4  # 가치 폭주 완화
    eps_clip: float = 0.2
    entropy_coef: float = 0.03  # 탐색 강화
    value_coef: float = 0.1  # critic 가중치 축소
    max_grad_norm: float = 0.3  # 그래디언트 클리핑 완화
    mini_batch: int = 4
    train_epochs: int = 4

    # 신호 페이즈(phase) 선택 액션: 0~5 (tls.add.xml 기준 6개 phase)
    action_phases: tuple = (0, 1, 2, 3, 4, 5)
    action_interval: int = 3  # 너무 잦은 스위칭을 완화하며 반응성 확보


# 관측 설계
OBS_LANES_PER_TLS = {
    "J1": ["1_J1_0", "1_J1_1", "2_J1_0", "2_J1_1", "J2_J1_0", "J2_J1_1", "J4_J1_0", "J4_J1_1"],
    "J2": ["3_J2_0", "3_J2_1", "4_J2_0", "4_J2_1", "J1_J2_0", "J1_J2_1", "J3_J2_0", "J3_J2_1"],
    "J3": ["5_J3_0", "5_J3_1", "6_J3_0", "6_J3_1", "J2_J3_0", "J2_J3_1", "J4_J3_0", "J4_J3_1"],
    "J4": ["7_J4_0", "7_J4_1", "8_J4_0", "8_J4_1", "J1_J4_0", "J1_J4_1", "J3_J4_0", "J3_J4_1"],
}

# GUI/CLI 실행 파일 이름
SUMO_BIN = "sumo-gui.exe" if os.name == "nt" else "sumo-gui"
SUMO_BIN_CLI = "sumo.exe" if os.name == "nt" else "sumo"
