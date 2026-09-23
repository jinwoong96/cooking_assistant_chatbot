# 할 일

## 음성 기능 — 직접 확인 필요

TTS 샘플(Supertonic)은 채팅으로 보냈고 아래 폴더에도 있음 (임시 폴더라 지워질 수 있음):
`C:\Users\wlsdn\AppData\Local\Temp\claude\C--Project-Files-cook-chatbot\c0259840-5dfc-459d-8fcd-5be1e2a52e38\scratchpad\tts_samples`

### TTS
- [ ] **음질**: 괜찮은지. 별로면 MeloTTS로 교체 검토
- [ ] **모델**: supertonic-2(빠름, 13초 음성 합성에 약 0.6초) vs supertonic-3(약 2초)
  → 현재 기본값 `supertonic-2`, 바꾸려면 `.env`의 `TTS_MODEL`
- [ ] **음성**: F1~F5, M1~M5 중 선택 → 현재 기본값 `F1`, `.env`의 `TTS_VOICE`
- [ ] **숫자 읽기**: `_digits` 샘플에서 "51,927원", "2인분", "300g"이 자연스러운지.
  어색하면 음성 요약 템플릿(`agent/pipeline.py`의 `recipe_speech_summary`)에서
  숫자를 한글로 바꿔 넣도록 수정
- [ ] **실제 앱 음성**: `app_recipe_summary.wav`(레시피 템플릿 요약),
  `app_general_chat_reply.wav`(일반 대화 전체 읽기)가 괜찮은지

### STT (실제 마이크로)
지금까지는 TTS로 만든 음성으로만 검증함. 사람 목소리로는 아직 확인 안 됨.
- [ ] **브라우저 마이크**: 녹음 버튼 → 말하기 → 말 끝나면 자동 전송되는지, 중지 버튼으로 멈추는지
- [ ] **PC 마이크**: 입력 장치 선택 → 듣기 시작 → 자동 전송 → 중지
- [ ] **답변 중 무시**: 봇이 답을 만드는 동안 / 음성 답변이 재생되는 동안 말한 게 무시되는지
  (스피커 소리가 다시 인식돼 봇이 혼자 대화하면 안 됨)
- [ ] **기본 모델**: 순환 테스트에선 medium이 제일 정확했음(turbo는 "김치찌개"를 "김칫"으로).
  실제 목소리로도 그런지 확인 → 기본값은 `.env`의 `STT_MODEL`
- [ ] 문장 중간에 쉬면 두 메시지로 쪼개지는지 (쪼개진 게 턴 시작 전에 모이면 하나로 합쳐 보내긴 함)
