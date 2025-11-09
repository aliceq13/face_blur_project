# -*- coding: utf-8 -*-
"""
비디오 앱 URL 라우팅
DRF Router를 사용하여 RESTful API 엔드포인트 자동 생성
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# DRF Router 설정
# Router는 ViewSet에서 자동으로 URL 패턴을 생성합니다
router = DefaultRouter()

# ViewSet 등록
router.register(r'videos', views.VideoViewSet, basename='video')
router.register(r'faces', views.FaceViewSet, basename='face')
router.register(r'jobs', views.ProcessingJobViewSet, basename='processingjob')

# URL 패턴
# /api/videos/ - 비디오 목록, 업로드
# /api/videos/{id}/ - 비디오 상세, 수정, 삭제
# /api/videos/{id}/start_processing/ - 영상 처리 시작
# /api/faces/ - 얼굴 목록
# /api/faces/{id}/ - 얼굴 상세, 블러 선택 변경
# /api/jobs/ - 처리 작업 목록
# /api/jobs/{id}/ - 처리 작업 상세

urlpatterns = [
    path('', include(router.urls)),
]
