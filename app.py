from flask import Flask, request, jsonify
from medkit.core.audio import MemoryAudioBuffer, AudioDocument
from medkit.audio.segmentation.pa_speaker_detector import PASpeakerDetector
from medkit.audio.transcription.sb_transcriber import SBTranscriber
import librosa
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)  # Capture tous les messages de niveau DEBUG et supérieur

app = Flask(__name__)

# Initialization of the speaker detector and transcriber
speaker_detector = PASpeakerDetector(
    model="medkit/simsamu-diarization",
    device=-1,
    segmentation_batch_size=10,
    embedding_batch_size=10,
    output_label="speaker"
)

transcriber = SBTranscriber(
    model="medkit/simsamu-transcription",
    needs_decoder=False,
    output_label="transcription",
    device=-1,
    batch_size=10,
)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    print("Received a request")
    if 'file' not in request.files:
        print("Error: No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected")
        return jsonify({"error": "No selected file"}), 400

    try:
        y, sr = librosa.load(file, sr=None)
        print("File loaded successfully")
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        if y_resampled.ndim == 1:
            y_resampled = y_resampled[np.newaxis, :]  # Add a dimension for channels
        print("Audio resampled")
        
        audio_buffer = MemoryAudioBuffer(signal=y_resampled, sample_rate=16000)
        audio_doc = AudioDocument(audio=audio_buffer)

        print("Starting speech segment detection")
        speech_segments = speaker_detector.run([audio_doc.raw_segment])
        print("Speech segments detected:", speech_segments)

        try:
            print("Starting transcription process")
            transcriptions = transcriber.run(speech_segments)
            print("Transcription completed:", transcriptions)
        except Exception as e:
            app.logger.exception("An error occurred during transcription: %s", str(e))
            return jsonify({"error": str(e)}), 500


        # Prepare the response
        transcription_results = []
        for speech_seg in speech_segments:
            transcription_attrs = speech_seg.attrs.get(label="transcription")
            for attr in transcription_attrs:  # Iterate over all matching attributes
                print(speech_seg.span.start, speech_seg.span.end, attr.value)
                transcription_results.append({
                    "start": speech_seg.span.start,
                    "end": speech_seg.span.end,
                    "text": attr.value
                })

        return jsonify({"transcriptions": transcription_results}), 200
    except Exception as e:
        print("An error occurred:", str(e))
        app.logger.error('Erreur lors du traitement : %s', str(e))
        return jsonify({"error": str(e)}), 500

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)