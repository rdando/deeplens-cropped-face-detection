import os
import logging
import boto3
import json
import base64

# Set Logging Level
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """
    Decode a Base64 encoded JPEG image and save to an S3 Bucket
    """
    logger.info('Event JSON: {0}'.format(json.dumps(event)))

    # Convert image back to binary
    jpg_original = base64.b64decode(event['image_string'])

    # Save image to S3 with the timestamp as the name
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Body=jpg_original,
        Bucket=os.environ["DETECTED_FACES_BUCKET"],
        Key='{}.jpg'.format(event['unix_time']),
    )
