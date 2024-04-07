# Download Default image python 3.8
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 라이브러리 파일 복사 및 설치
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Application 코드 복사
COPY train.py /app

# 스크립트 실행
CMD ["python", "./train.py"]