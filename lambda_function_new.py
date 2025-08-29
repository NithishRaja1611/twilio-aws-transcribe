import json
import os
import time
import uuid
import boto3
import requests
from urllib.parse import unquote_plus
from openai import OpenAI
from dotenv import load_dotenv   # ✅ for local testing

# Load .env file locally (ignored in AWS if no file present)
load_dotenv()

# AWS clients
s3 = boto3.client("s3")
transcribe = boto3.client("transcribe")

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def lambda_handler(event, context):
    try:
        # 1. Get S3 event info (Twilio uploaded recording)
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = unquote_plus(event['Records'][0]['s3']['object']['key'])
        file_uri = f"s3://{bucket}/{key}"

        print(f"📂 Received audio file: {file_uri}")

        # 2. Start AWS Transcribe job
        job_name = f"twilio-{uuid.uuid4().hex[:8]}"
        media_format = key.split(".")[-1].lower()

        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat=media_format,
            LanguageCode="en-US",
            OutputBucketName=bucket,
            OutputKey=f"transcripts/{job_name}.json"
        )

        print(f"▶️ Started Transcribe job: {job_name}")

        # 3. Poll until transcription completes (max ~5 min)
        for _ in range(30):  # 30 x 10s = 5 minutes
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']

            if job_status in ['COMPLETED', 'FAILED']:
                break

            print("⏳ Waiting for transcription...")
            time.sleep(10)

        if job_status == "FAILED":
            raise Exception("Transcription failed")

        print("✅ Transcription completed")

        # 4. Get transcript JSON from S3
        transcript_key = f"transcripts/{job_name}.json"
        obj = s3.get_object(Bucket=bucket, Key=transcript_key)
        transcript_json = json.loads(obj["Body"].read())

        transcript_text = transcript_json['results']['transcripts'][0]['transcript']
        print("📜 Transcript extracted. Sample:", transcript_text[:150])

        # 5. Summarize with OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize medical consultations clearly and concisely."},
                {"role": "user", "content": transcript_text}
            ],
            max_tokens=300
        )
        summary = response.choices[0].message.content

        # 6. Save only the summary as plain text (.txt) in S3
        summary_key = f"summaries/{job_name}_summary.txt"
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=summary.encode("utf-8"),
            ContentType="text/plain"
        )

        print(f"✅ Summary saved: s3://{bucket}/{summary_key}")

        return {"status": "success", "summary_file": f"s3://{bucket}/{summary_key}"}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}


# 🔹 Local testing only (won’t run in AWS Lambda)
if __name__ == "__main__":
    with open("test_event.json") as f:
        event = json.load(f)
    result = lambda_handler(event, None)
    print("Local run result:", result)
