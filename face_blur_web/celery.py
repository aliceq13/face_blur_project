# -*- coding: utf-8 -*-
"""
Celery 설정 파일

이 파일은 Celery 애플리케이션을 초기화하고 Django 설정과 통합합니다.

Celery란?
- 분산 작업 큐 시스템
- 오래 걸리는 작업(영상 처리, 얼굴 분석)을 백그라운드에서 비동기적으로 실행
- Redis를 메시지 브로커로 사용하여 작업 전달
"""

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Django 설정 모듈 지정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')

# Celery 앱 생성
app = Celery('face_blur_web')

# Django settings.py에서 CELERY_ 접두사가 붙은 설정 자동 로드
# 예: CELERY_BROKER_URL, CELERY_RESULT_BACKEND 등
app.config_from_object('django.conf:settings', namespace='CELERY')

# Django 앱들에서 tasks.py 파일을 자동으로 찾아 등록
# apps/videos/tasks.py, apps/accounts/tasks.py 등
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """
    디버깅용 테스트 태스크
    Celery 연결 테스트 시 사용:

    >>> from face_blur_web.celery import debug_task
    >>> debug_task.delay()
    """
    print(f'Request: {self.request!r}')
