from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import boto3
import uuid
import time
import os
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription API", version="1.0.0")

# Initialize AWS clients
try:
    transcribe_client = boto3.client('transcribe')
    s3_client = boto3.client('s3')
except NoCredentialsError:
    logger.error("AWS credentials not found. Please configure your credentials.")
    raise

# Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-transcribe-bucket')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.flac', '.ogg', '.amr', '.webm'}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribe an audio file using Amazon Transcribe
    
    Args:
        audio_file: Audio file to transcribe (supported formats: mp3, mp4, wav, flac, ogg, amr, webm)
    
    Returns:
        JSON response with transcribed text
    """
    
    # Validate file format
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Generate unique job name and S3 key
    job_name = f"transcribe-job-{uuid.uuid4()}"
    s3_key = f"audio-files/{job_name}{file_extension}"
    
    try:
        # Upload file to S3
        logger.info(f"Uploading file to S3: {s3_key}")
        s3_client.upload_fileobj(
            audio_file.file,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': audio_file.content_type}
        )
        
        # Construct S3 URI
        media_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        
        # Determine media format for Transcribe
        media_format = file_extension[1:]  # Remove the dot
        if media_format == 'mp4':
            media_format = 'mp4'
        elif media_format == 'ogg':
            media_format = 'ogg'
        elif media_format == 'webm':
            media_format = 'webm'
        
        # Start transcription job
        logger.info(f"Starting transcription job: {job_name}")
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=media_format,
            LanguageCode='en-US',  # You can make this configurable
            Settings={
                'ShowSpeakerLabels': False,
                'MaxSpeakerLabels': 2
            }
        )
        
        # Poll for job completion
        max_wait_time = 300  # 5 minutes
        wait_interval = 5    # 5 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            response = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            if status == 'COMPLETED':
                # Get the transcript
                transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                
                # Download and parse transcript
                import requests
                transcript_response = requests.get(transcript_uri)
                transcript_data = transcript_response.json()
                
                # Extract the transcribed text
                transcribed_text = transcript_data['results']['transcripts'][0]['transcript']
                
                # Clean up: delete the audio file from S3
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    logger.info(f"Deleted temporary file: {s3_key}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "transcribed_text": transcribed_text,
                        "job_name": job_name,
                        "confidence_scores": [
                            item.get('alternatives', [{}])[0].get('confidence', 'N/A')
                            for item in transcript_data['results']['items']
                            if item['type'] == 'pronunciation'
                        ]
                    }
                )
            
            elif status == 'FAILED':
                failure_reason = response['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {failure_reason}"
                )
            
            # Wait before checking again
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            logger.info(f"Transcription job status: {status}. Elapsed time: {elapsed_time}s")
        
        # Timeout reached
        raise HTTPException(
            status_code=408,
            detail="Transcription job timed out. Please try again with a shorter audio file."
        )
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'NoSuchBucket':
            raise HTTPException(
                status_code=500,
                detail=f"S3 bucket '{S3_BUCKET_NAME}' does not exist. Please create it first."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"AWS error ({error_code}): {error_message}"
            )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
    finally:
        # Clean up transcription job
        try:
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            logger.info(f"Deleted transcription job: {job_name}")
        except Exception as e:
            logger.warning(f"Failed to delete transcription job: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Audio Transcription API"}

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "supported_formats": list(SUPPORTED_FORMATS),
        "note": "Maximum file size depends on your S3 and Transcribe limits"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
