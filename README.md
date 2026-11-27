# ESMFold on Furiosa RNGD

아미노산 서열로 단백질 구조를 예측하고 브라우저에서 3차원으로 확인하는 웹 서비스입니다. Meta AI가 공개한 **ESMFold v1**을 Hugging Face Transformers로 불러옵니다. 이 저장소는 기존 모델을 FuriosaAI RNGD에서 실행하는 코드와 웹 인터페이스를 제공합니다.

단일 체인 **3–50 aa**를 입력할 수 있습니다. 표준 아미노산 20종과 `X`를 지원합니다. 예측이 끝나면 잔기별 신뢰도(pLDDT)에 따라 구조를 색칠하고 추론 시간, PDB 파일, 원자 좌표 CSV를 제공합니다.

## 추론 구성

ESM-2 3B 백본 36층을 RNGD 두 장에 18층씩 배치해 순서대로 실행합니다. 각 카드는 8 PE를 사용합니다. 백본 입력을 52토큰으로 패딩하고 패딩 위치를 마스킹하여 지원 범위의 서열이 컴파일 캐시를 공유합니다. Folding Trunk와 Structure Module은 CPU FP32에서 실행합니다. 웹 서비스는 한 번에 한 요청을 처리하며 기본 추가 recycle 횟수는 3입니다.

## 설치와 실행

검증 환경은 Linux, Python 3.12, RNGD 두 장, PyTorch 2.10.0+cpu, furiosa-torch 및 furiosa-kernels 2026.3.0, Transformers 4.57.6입니다. 먼저 호스트에 RNGD 드라이버와 도구를 설치하고 `furiosa-smi info`로 장치 두 대를 확인하세요. Python 환경에는 해당 버전의 Furiosa 패키지와 CPU용 PyTorch wheel을 설치할 수 있는 패키지 소스가 필요합니다. [Furiosa 개발자 문서](https://developer.furiosa.ai/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# SDK 배포 환경에서 위 버전의 Furiosa 패키지와 CPU PyTorch를 먼저 설치합니다.
python -m pip install -r requirements.lock.txt
./download_model.sh
./service/run_service.sh start
./service/run_service.sh status
```

모델은 고정 리비전 `75a3841ee059df2bf4d56688166c8fb459ddd97a`에서 다운로드합니다. 가중치 파일은 약 8.44 GB입니다. 첫 기동에는 백본 컴파일 시간이 걸리며 생성된 캐시는 `cache/service/`에 저장됩니다. 로그는 `logs/service.log`에서 확인합니다. 상태가 `ready`가 되면 서버에서 <http://127.0.0.1:8090>을 여세요. 원격 PC에서는 다음 명령의 사용자와 호스트를 바꾸고, SSH 연결을 유지한 채 같은 주소를 엽니다.

```bash
ssh -N -L 8090:127.0.0.1:8090 USER@HOST
```

## API 예제

```bash
curl -sS http://127.0.0.1:8090/api/status
curl -sS http://127.0.0.1:8090/api/fold -H 'Content-Type: application/json' \
  -d '{"sequence":"MQIFVKTLTGKTITLEV"}'
# 응답의 id를 JOB_ID에 넣고, state가 done이 된 뒤 구조를 저장합니다.
JOB_ID=여기에_ID_입력
curl -sS "http://127.0.0.1:8090/api/jobs/$JOB_ID"
curl -f "http://127.0.0.1:8090/api/jobs/$JOB_ID/pdb" -o prediction.pdb
```

종료는 `./service/run_service.sh stop`, 수치 호환성 테스트는 `.venv/bin/python -m unittest -v test_rngd_compat`로 실행합니다. CPU에서 웹 동작만 확인할 때는 `./service/run_service.sh start --cpu`를 사용할 수 있습니다. 서비스에는 사용자 인증이 없으며 기본 연결 주소는 루프백입니다.

## 모델 출처와 인용

- Lin et al. (2023), “Evolutionary-scale prediction of atomic-level protein structure with a language model”, *Science*. [논문 DOI](https://doi.org/10.1126/science.ade2574)
- ESMFold v1 및 ESM-2의 원 개발자는 Meta AI의 FAIR 연구팀입니다. [원본 ESM 저장소](https://github.com/facebookresearch/esm) · [모델 카드 및 가중치](https://huggingface.co/facebook/esmfold_v1)
- 모델 구현은 Hugging Face Transformers의 ESMFold 포트를 사용합니다. 3D 뷰어는 3Dmol.js 2.5.3입니다. [Transformers](https://github.com/huggingface/transformers) · [3Dmol.js](https://github.com/3dmol/3Dmol.js)

상위 프로젝트의 라이선스는 [notices](notices/)에, 번들에 포함된 라이브러리 고지는 [뷰어 배포 고지문](service/static/vendor/3Dmol-min.js.LICENSE.txt)에 있습니다. 모델 가중치와 Furiosa SDK는 별도로 설치합니다.
