# 프로젝트 실행 방법
git clone
conda create -n sumo_python
conda activate sumo_python
pip intall -r requirement.txt
cd sumo_python
python main.py
실행된 sumo의 좌측 상단에 있는 화살표 버튼을 눌러 시뮬레이션 시작

main.py 의 218, 219 line을 하나씩 주석처리하면서 고정 신호 주기와 mappo 모델의 변동 신호 주기의 결과를 알 수 있습니다.


이미 학습된 모델은 models directory에 있으며, 직접 학습시키고 싶으면 python train_mappo.py 명령어를 실행하여 직접 학습시킬 수 있습니다.


학습 결과는 logs directory에 csv 파일로 저장됩니다.
