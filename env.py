import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import traci

import config as cfg


class SumoEnv:
    """
    MAPPO용 SUMO 다중 신호 제어 환경.
    - 에이전트: TLS (cfg.TLS_IDS)
    - 관측: 각 TLS에 연결된 진입 차로 대기열 길이(대수 합계)
    - 행동: 현재 페이즈 지속시간을 action_durations 중 하나로 설정
    - 보상: -전체 대기시간 합계(episode 단계마다)
    """

    def __init__(self, use_gui: bool = cfg.USE_GUI_DEFAULT, step_length: float = cfg.DEFAULT_STEP_LENGTH, suppress_warnings: bool = True):
        self.use_gui = use_gui
        self.step_length = step_length
        self.suppress_warnings = suppress_warnings
        self.tls_ids = cfg.TLS_IDS
        self.action_candidates = cfg.MAPPOConfig.action_phases  # 선택 가능한 페이즈 인덱스
        self.action_interval = cfg.MAPPOConfig.action_interval  # 선택한 페이즈를 유지할 스텝 수
        self.net_file = cfg.NET_FILE
        self.route_file = cfg.ROUTE_FILE
        self.tls_file = cfg.TLS_FILE
        self.sumo_root = cfg.SUMO_ROOT
        self.site_root = Path(__file__).resolve().parent
        self._proc = None
        self._step = 0
        self._null_fh: Optional[object] = None
        self._last_teleports = 0
        self.delay_accum = 0.0
        self._queue_sum: Dict[str, float] = {}
        self._queue_steps: int = 0

    def _ensure_null_fh(self) -> Optional[object]:
        if not self.suppress_warnings:
            return None
        if self._null_fh is None or self._null_fh.closed:
            self._null_fh = open(os.devnull, "w")
        return self._null_fh

    # ---------- SUMO 실행 ----------
    def _build_cmd(self) -> List[str]:
        bin_name = cfg.SUMO_BIN if self.use_gui else cfg.SUMO_BIN_CLI
        exe = self.sumo_root / "bin" / bin_name
        if not exe.exists():
            raise FileNotFoundError(f"SUMO 실행 파일을 찾을 수 없습니다: {exe}")
        for f in [self.net_file, self.route_file, self.tls_file]:
            if not f.exists():
                raise FileNotFoundError(f"필수 입력 파일이 없습니다: {f}")
        cmd = [
            str(exe),
            "-n",
            str(self.net_file),
            "-r",
            str(self.route_file),
            "-a",
            str(self.tls_file),
            "--step-length",
            str(self.step_length),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        if self.use_gui:
            cmd += ["--delay", "100"]
        return cmd

    def reset(self) -> Dict[str, np.ndarray]:
        """SUMO 재시작 후 초기 관측을 반환."""
        self.close()
        cmd = self._build_cmd()
        null_fh = self._ensure_null_fh()
        traci.start(cmd, stdout=null_fh)
        self._step = 0
        # 텔레포트 카운트 초기화
        self._last_teleports = traci.simulation.getStartingTeleportNumber()
        self.delay_accum = 0.0
        self._queue_sum.clear()
        self._queue_steps = 0
        return self._get_observations()

    def close(self) -> None:
        """기존 세션 종료."""
        if traci.isLoaded():
            traci.close(False)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        if self._null_fh:
            try:
                self._null_fh.close()
            except Exception:
                pass
            self._null_fh = None
        self._proc = None

    # ---------- 관측/보상 ----------
    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for tls_id in self.tls_ids:
            lanes = cfg.OBS_LANES_PER_TLS[tls_id]
            waitings = [traci.lane.getWaitingTime(lane) for lane in lanes]
            # 정규화: 초 단위 대기시간을 0~1 스케일링
            vec = np.array(waitings, dtype=np.float32) / 300.0
            obs[tls_id] = vec
        return obs

    def _compute_reward(self) -> float:
        """보상: 평균 대기시간 + 텔레포트/긴급정지 패널티를 최소화."""
        total_wait = 0.0
        for lane in traci.lane.getIDList():
            total_wait += traci.lane.getWaitingTime(lane)

        veh_count = max(1, len(traci.vehicle.getIDList()))
        avg_wait = total_wait / veh_count  # 초 단위

        # 텔레포트 증가량 패널티
        cur_tel = traci.simulation.getStartingTeleportNumber()
        tel_delta = cur_tel - self._last_teleports
        self._last_teleports = cur_tel

        # 긴급 정지/급제동 패널티 (지원되는 경우)
        emergency_delta = 0
        try:
            emergencies = traci.simulation.getEmergencyStoppingVehiclesIDList()
            emergency_delta = len(emergencies)
        except traci.TraCIException:
            emergency_delta = 0

        # 스케일: 평균대기/100 + 0.1*텔레포트 + 0.05*긴급정지
        reward = -((avg_wait / 100.0) + 0.1 * tel_delta + 0.05 * emergency_delta)
        return reward

    # ---------- 액션 적용 ----------
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool]:
        """
        actions: tls_id -> action index (phase 선택)
        반환: obs, reward_dict(공유 보상), done
        """
        # 페이즈 선택 및 유지 시간 설정
        for tls_id, act_idx in actions.items():
            try:
                phase = self.action_candidates[act_idx]
                traci.trafficlight.setPhase(tls_id, phase)
                traci.trafficlight.setPhaseDuration(tls_id, max(1.0, self.action_interval * self.step_length))
            except traci.TraCIException:
                # 존재하지 않는 경우 무시
                pass

        # action_interval 동안 step 진행
        for _ in range(self.action_interval):
            # 현재 step의 총 대기시간을 누적하여 총 delay(초) 추정
            try:
                wait_now = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
                self.delay_accum += wait_now * self.step_length

                # 차로별 대기열(정지 차량 수) 합산
                for lane in traci.lane.getIDList():
                    q = traci.lane.getLastStepHaltingNumber(lane)
                    self._queue_sum[lane] = self._queue_sum.get(lane, 0.0) + q
                self._queue_steps += 1
            except traci.TraCIException:
                pass
            traci.simulationStep()
            self._step += 1

        obs = self._get_observations()
        reward = self._compute_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0
        # 모든 에이전트에 동일 보상 공유
        rewards = {tid: reward for tid in self.tls_ids}
        return obs, rewards, done

    def get_avg_queue_lengths(self) -> Dict[str, float]:
        """각 차로별 평균 대기열 길이(정지 차량 수)."""
        if self._queue_steps == 0:
            return {lane: 0.0 for lane in traci.lane.getIDList()}
        return {lane: total / self._queue_steps for lane, total in self._queue_sum.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
