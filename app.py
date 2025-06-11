# import os
# import re
# import io
# import logging
# import requests
# import tempfile
# import subprocess
# import cv2
# from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# import yt_dlp
# from whisper import load_model, DecodingOptions
# import spacy
# import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import BertTokenizer, BertModel
# import torch
# from ultralytics import YOLO

# app = Flask(__name__)
# app.secret_key = os.urandom(24)
# app.jinja_env.filters['format_time'] = lambda s: format_time(s)

# # Configuration
# YOUTUBE_API_KEY = "AIzaSyDeorAZqvDwHV3xSG04rAbxnR6jtDB9mzM"
# os.makedirs('downloads', exist_ok=True)
# os.makedirs('static/images', exist_ok=True)

# # Initialize models
# whisper_model = None
# nlp = spacy.load("en_core_web_md")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# yolo_model = YOLO("yolov9c.pt")

# def format_time(seconds):
#     hours, remainder = divmod(seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#     return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# def initialize_whisper():
#     global whisper_model
#     if whisper_model is None:
#         whisper_model = load_model("base.en")
#     return DecodingOptions(fp16=False, language="en")

# def search_youtube(query):
#     try:
#         youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
#         response = youtube.search().list(
#             q=query,
#             part="snippet",
#             type="video",
#             maxResults=5,
#         ).execute()
#         return [{
#             "title": item["snippet"]["title"],
#             "video_id": item["id"]["videoId"],
#             "description": item["snippet"]["description"],
#             "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
#         } for item in response.get("items", [])]
#     except Exception as e:
#         logging.error(f"Search error: {e}", exc_info=True)
#         return []

# def download_video(video_url):
#     try:
#         video_id = video_url.split("v=")[-1].split('&')[0]
#         file_path = f"downloads/{video_id}.mp4"
        
#         if os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#             except Exception as e:
#                 logging.error(f"Error removing existing file: {e}")
        
#         ydl_opts = {
#             "format": "mp4",
#             "outtmpl": file_path,
#             "quiet": True,
#         }
        
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_id])
            
#         return file_path
        
#     except Exception as e:
#         logging.error(f"Download error: {e}", exc_info=True)
#         raise

# def transcribe_video(file_path):
#     try:
#         global whisper_model
#         options = initialize_whisper()
        
#         if whisper_model is None:
#             whisper_model = load_model("base.en")
            
#         result = whisper_model.transcribe(file_path, **options.__dict__, verbose=False)
#         return result["segments"]
        
#     except Exception as e:
#         logging.error(f"Transcription error: {e}", exc_info=True)
#         return []

# def extract_entities_and_relationships(text, graph):
#     try:
#         doc = nlp(text)
#         entities = [(ent.text, ent.label_) for ent in doc.ents]
#         for i, (entity_text, entity_label) in enumerate(entities):
#             graph.add_node(entity_text, label=entity_label)
#             if i > 0:
#                 graph.add_edge(entities[i-1][0], entity_text)
#         return entities
#     except Exception as e:
#         logging.error(f"Entity extraction error: {e}", exc_info=True)
#         return []

# def extract_bert_embeddings(text):
#     try:
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     except Exception as e:
#         logging.error(f"BERT embedding error: {e}", exc_info=True)
#         return np.zeros(768)

# def build_cognitive_graph(videos, transcripts):
#     try:
#         graph = nx.Graph()
#         for video, transcript in zip(videos, transcripts):
#             video_title = video['title']
#             transcript_text = " ".join([seg['text'] for seg in transcript])
            
#             graph.add_node(video_title, type="video")
#             extract_entities_and_relationships(video['description'], graph)
#             extract_entities_and_relationships(transcript_text, graph)
#         return graph
#     except Exception as e:
#         logging.error(f"Graph build error: {e}", exc_info=True)
#         return nx.Graph()

# def rank_videos(query, graph):
#     try:
#         query_embed = extract_bert_embeddings(query)
#         scores = []
#         for node in graph.nodes:
#             if graph.nodes[node].get("type") == "video":
#                 node_embed = extract_bert_embeddings(node)
#                 similarity = cosine_similarity([query_embed], [node_embed])[0][0]
#                 scores.append((node, float(similarity)))
#         return sorted(enumerate(scores), key=lambda x: x[1][1], reverse=True)
#     except Exception as e:
#         logging.error(f"Ranking error: {e}", exc_info=True)
#         return []

# def visualize_graph(top_ranked_videos, graph):
#     try:
#         plt.clf()
#         subgraph = nx.Graph()
        
#         for idx, (_, score) in top_ranked_videos[:3]:
#             if not session.get('results') or idx >= len(session['results']):
#                 continue
#             video = session['results'][idx][0]
#             video_title = video['title']
#             subgraph.add_node(video_title, type='video', score=float(score))
            
#             doc = nlp(video['description'])
#             entities = [(ent.text, ent.label_) for ent in doc.ents]
            
#             for entity_text, entity_label in entities:
#                 subgraph.add_node(entity_text, type=entity_label)
#                 subgraph.add_edge(video_title, entity_text)
        
#         pos = nx.spring_layout(subgraph)
#         node_colors = ['#FF9999' if subgraph.nodes[n].get('type') == 'video' else '#99FF99' 
#                       for n in subgraph.nodes]
        
#         plt.figure(figsize=(10, 8))
#         nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, 
#                node_size=2500, font_size=10, font_weight='bold')
#         plt.savefig('static/images/graph.png', bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         logging.error(f"Visualization error: {e}", exc_info=True)

# # Enhanced video processing functions
# def get_video_fps(video_path):
#     video = cv2.VideoCapture(video_path)
#     fps = video.get(cv2.CAP_PROP_FPS)
#     video.release()
#     return fps

# def extract_frames(video_path, frame_folder):
#     os.makedirs(frame_folder, exist_ok=True)
#     video = cv2.VideoCapture(video_path)
#     frame_count = 0
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         path = os.path.join(frame_folder, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(path, frame)
#         frame_count += 1
#     video.release()
#     return frame_count

# def upscale_frames_executable(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
#     exe_path = os.path.join("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan")

#     for file in input_files:
#         input_path = os.path.join(input_folder, file)
#         output_path = os.path.join(output_folder, file)
#         command = [
#             exe_path,
#             "-i", input_path,
#             "-o", output_path,
#             "-n", "realesrgan-x4plus"
#         ]
#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# def create_video_from_frames(frame_folder, output_video, fps=30):
#     files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
#     if not files:
#         return False

#     frame_path = os.path.join(frame_folder, files[0])
#     frame = cv2.imread(frame_path)
#     height, width, _ = frame.shape

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#     for file in files:
#         img = cv2.imread(os.path.join(frame_folder, file))
#         out.write(img)

#     out.release()
#     return True

# def merge_audio(original_video, upscaled_video, final_output):
#     command = [
#         "ffmpeg", "-y",
#         "-i", upscaled_video,
#         "-i", original_video,
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-map", "0:v:0",
#         "-map", "1:a:0",
#         "-shortest",
#         final_output
#     ]
#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return os.path.exists(final_output)

# @app.route('/enhance', methods=['POST'])
# def enhance_video():
#     video_id = request.form.get('video_id')
#     session['enhance_progress'] = 0
#     session['enhance_logs'] = ["1/5: Starting video enhancement..."]
#     session.pop('progress', None)
#     session.pop('logs', None)
    
#     try:
#         # Step 1: Download original video
#         session['enhance_logs'].append("2/5: Downloading original video...")
#         session['enhance_progress'] = 20
#         video_url = f"https://www.youtube.com/watch?v={video_id}"
#         original_path = download_video(video_url)
        
#         # Create directories
#         video_dir = os.path.join('downloads', video_id)
#         os.makedirs(video_dir, exist_ok=True)
#         frame_folder = os.path.join(video_dir, 'frames')
#         upscaled_folder = os.path.join(video_dir, 'upscaled_frames')
#         enhanced_path = os.path.join(video_dir, 'enhanced.mp4')
#         final_path = os.path.join(video_dir, 'enhanced_with_audio.mp4')
        
#         # Step 2: Extract frames
#         session['enhance_logs'].append("3/5: Extracting frames...")
#         session['enhance_progress'] = 40
#         total_frames = extract_frames(original_path, frame_folder)
        
#         # Step 3: Upscale frames
#         session['enhance_logs'].append("4/5: Upscaling frames (this may take several minutes)...")
#         session['enhance_progress'] = 60
#         upscale_frames_executable(frame_folder, upscaled_folder)
        
#         # Step 4: Create video
#         session['enhance_logs'].append("5/5: Creating enhanced video...")
#         session['enhance_progress'] = 80
#         fps = get_video_fps(original_path)
#         create_success = create_video_from_frames(upscaled_folder, enhanced_path, fps)
#         if not create_success:
#             raise Exception("Failed to create video from frames")
        
#         # Step 5: Merge audio
#         merge_success = merge_audio(original_path, enhanced_path, final_path)
#         if not merge_success:
#             raise Exception("Audio merge failed")
        
#         session['enhance_progress'] = 100
#         session['enhance_logs'].append("✅ Enhancement complete!")
#         return jsonify({"success": True, "enhanced_path": final_path})
    
#     except Exception as e:
#         logging.error(f"Enhancement error: {e}", exc_info=True)
#         return jsonify({"error": str(e)}), 500

# @app.route('/download-enhanced/<video_id>')
# def download_enhanced(video_id):
#     video_dir = os.path.join('downloads', video_id)
#     enhanced_path = os.path.join(video_dir, 'enhanced_with_audio.mp4')
#     if os.path.exists(enhanced_path):
#         return send_file(enhanced_path, as_attachment=True, download_name=f"enhanced_{video_id}.mp4")
#     return "Enhanced video not found", 404

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         session.clear()
#         query = request.form.get('query', '').strip()
        
#         try:
#             if not query:
#                 raise ValueError("Empty search query")
                
#             results = search_youtube(query)
#             if not results:
#                 raise ValueError("No videos found")
            
#             transcripts = []
#             for vid in results:
#                 try:
#                     vid_url = f"https://youtube.com/watch?v={vid['video_id']}"
#                     file_path = download_video(vid_url)
#                     segments = transcribe_video(file_path)
#                     transcripts.append(segments)
#                 except Exception as e:
#                     logging.error(f"Video processing failed: {e}", exc_info=True)
#                     continue

#             cognitive_graph = build_cognitive_graph(results, transcripts)
#             ranked = rank_videos(query, cognitive_graph)
            
#             session['results'] = [(results[idx], float(score)) 
#                                 for idx, (_, score) in ranked]
            
#             visualize_graph(ranked, cognitive_graph)
#             return redirect(url_for('results'))
            
#         except Exception as e:
#             logging.error(f"Search failed: {e}", exc_info=True)
#             return render_template('index.html', 
#                                  error=f"Search failed: {str(e)}")
    
#     return render_template('index.html')

# @app.route('/quick-search', methods=['POST'])
# def quick_search():
#     try:
#         session.clear()
#         query = request.form.get('query', '').strip()
#         if not query:
#             return redirect(url_for('index'))
        
#         results = search_youtube(query)
#         if not results:
#             return redirect(url_for('index'))
            
#         session['quick_results'] = [result['video_id'] for result in results]
#         return redirect(url_for('video', vid_id=results[0]['video_id']))
        
#     except Exception as e:
#         logging.error(f"Quick search failed: {e}", exc_info=True)
#         return redirect(url_for('index'))

# @app.route('/suggest', methods=['GET'])
# def fetch_youtube_suggestions():
#     try:
#         query = request.args.get('q', '')
#         if not query:
#             return jsonify([])

#         response = requests.get(
#             "https://suggestqueries.google.com/complete/search",
#             params={
#                 "client": "firefox",
#                 "ds": "yt",
#                 "q": query,
#                 "hl": "en"
#             },
#             headers={
#                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
#             },
#             timeout=2
#         )
#         if response.status_code == 200:
#             suggestions = response.json()[1]
#             return jsonify(suggestions[:5]), 200, {
#                 'Content-Type': 'application/json',
#                 'Access-Control-Allow-Origin': '*'
#             }
#         return jsonify([])
#     except Exception as e:
#         logging.error(f"Suggestions error: {e}", exc_info=True)
#         return jsonify([])

# @app.route('/results')
# def results():
#     return render_template('results.html', 
#                          results=session.get('results', []),
#                          graph_url='static/images/graph.png')

# @app.route('/video/<vid_id>')
# def video(vid_id):
#     try:
#         video = next((v for v, _ in session.get('results', []) if v['video_id'] == vid_id), None)
#         if not video:
#             return redirect(url_for('index'))
        
#         vid_url = f"https://youtube.com/watch?v={vid_id}"
#         file_path = download_video(vid_url)
#         segments = transcribe_video(file_path)
        
#         return render_template('video.html', 
#                             video=video,
#                             segments=segments,
#                             current_time=request.args.get('t', 0))
    
#     except Exception as e:
#         logging.error(f"Video load failed: {e}", exc_info=True)
#         return redirect(url_for('index'))

# @app.route('/play_video/<video_id>')
# def play_video(video_id):
#     try:
#         current_time = request.args.get('t', 0, type=float)
#         return render_template('video_player.html', 
#                              video_id=video_id,
#                              current_time=current_time)
#     except Exception as e:
#         logging.error(f"Play video failed: {e}", exc_info=True)
#         return redirect(url_for('index'))

# @app.route('/download/<vid_id>')
# def download(vid_id):
#     try:
#         video = next((v for v, _ in session.get('results', []) if v['video_id'] == vid_id), None)
#         if not video:
#             return redirect(url_for('index'))
        
#         segments = transcribe_video(f"downloads/{vid_id}.mp4")
#         srt = []
#         for i, seg in enumerate(segments):
#             srt.append(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n")
        
#         return send_file(
#             io.BytesIO("".join(srt).encode()),
#             mimetype='text/plain',
#             download_name=f"{vid_id}_transcript.srt",
#             as_attachment=True
#         )
    
#     except Exception as e:
#         logging.error(f"Download failed: {e}", exc_info=True)
#         return redirect(url_for('index'))

# @app.route('/analyze', methods=['POST'])
# def analyze_video():
#     video_id = request.form.get('video_id')
#     session['progress'] = 0
#     session['logs'] = ["1/4: Initializing analysis..."]
    
#     try:
#         session['progress'] = 25
#         video_path = download_video(f"https://www.youtube.com/watch?v={video_id}")

#         session['logs'].append("3/4: Analyzing video content...")
#         session['progress'] = 50
        
#         detections = []
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_interval = int(fps * 10)
        
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if frame_count % (total_frames // 10) == 0:
#                 progress = min(50 + int(40 * (frame_count / total_frames)), 90)
#                 session['progress'] = progress
#                 session['logs'].append(f"Processed {frame_count}/{total_frames} frames")
            
#             if frame_count % frame_interval == 0:
#                 results = yolo_model(frame)
#                 timestamp = frame_count / fps
#                 objects = list(set([
#                     yolo_model.names[int(box.cls)] 
#                     for box in results[0].boxes
#                 ]))
                
#                 if objects:
#                     detections.append({
#                         "timestamp": int(timestamp),
#                         "objects": objects,
#                         "embed_url": f"https://www.youtube.com/embed/{video_id}?start={int(timestamp)}&autoplay=1"
#                     })
            
#             frame_count += 1
        
#         session['progress'] = 100
#         session['logs'].append("4/4: Analysis complete!")
#         cap.release()
#         os.remove(video_path)
#         return jsonify(detections)
    
#     except Exception as e:
#         if 'video_path' in locals() and os.path.exists(video_path):
#             os.remove(video_path)
#         return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# @app.route('/progress')
# def get_progress():
#     return jsonify({
#         "progress": session.get('progress', 0),
#         "logs": session.get('logs', []),
#         "enhance_progress": session.get('enhance_progress', 0),
#         "enhance_logs": session.get('enhance_logs', [])
#     })

# @app.errorhandler(404)
# def not_found(e):
#     return render_template('error.html', error="Page not found"), 404

# @app.errorhandler(500)
# def server_error(e):
#     return render_template('error.html', error="Server error"), 500

# if __name__ == '__main__':
#     app.run(debug=False)

import os
import re
import io
import logging
import requests
import tempfile
import subprocess
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
from whisper import load_model, DecodingOptions
import spacy
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from ultralytics import YOLO
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer, GenerationConfig # Modified import
import torch

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.jinja_env.filters['format_time'] = lambda s: format_time(s)

# Configuration
YOUTUBE_API_KEY = "AIzaSyDeorAZqvDwHV3xSG04rAbxnR6jtDB9mzM"
os.makedirs('downloads', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Initialize models
whisper_model = None
nlp = spacy.load("en_core_web_md")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
yolo_model = YOLO("yolov9c.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BART model with explicit generation config
bart_model = BartForConditionalGeneration.from_pretrained(
    "models/bart_model",
    use_safetensors=True  # Add this since you're using model.safetensors
).to(device)

# Manually set missing generation parameters
bart_model.generation_config.early_stopping = True
bart_model.generation_config.max_length = 150
bart_model.generation_config.num_beams = 4

# Load tokenizer with special tokens mapping
bart_tokenizer = BartTokenizer.from_pretrained(
    "models/bart_model",
    eos_token="</s>",
    pad_token="<pad>"
) # Added BART tokenizer

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def initialize_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = load_model("base.en")
    return DecodingOptions(fp16=False, language="en")

def search_youtube(query):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=5,
        ).execute()
        return [{
            "title": item["snippet"]["title"],
            "video_id": item["id"]["videoId"],
            "description": item["snippet"]["description"],
            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
        } for item in response.get("items", [])]
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return []

def download_video(video_url):
    try:
        video_id = video_url.split("v=")[-1].split('&')[0]
        file_path = f"downloads/{video_id}.mp4"
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Error removing existing file: {e}")
        
        ydl_opts = {
            "format": "mp4",
            "outtmpl": file_path,
            "quiet": True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_id])
            
        return file_path
        
    except Exception as e:
        logging.error(f"Download error: {e}", exc_info=True)
        raise

def transcribe_video(file_path):
    try:
        global whisper_model
        options = initialize_whisper()
        
        if whisper_model is None:
            whisper_model = load_model("base.en")
            
        result = whisper_model.transcribe(file_path, **options.__dict__, verbose=False)
        return result["segments"]
        
    except Exception as e:
        logging.error(f"Transcription error: {e}", exc_info=True)
        return []

def extract_entities_and_relationships(text, graph):
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        for i, (entity_text, entity_label) in enumerate(entities):
            graph.add_node(entity_text, label=entity_label)
            if i > 0:
                graph.add_edge(entities[i-1][0], entity_text)
        return entities
    except Exception as e:
        logging.error(f"Entity extraction error: {e}", exc_info=True)
        return []

def extract_bert_embeddings(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        logging.error(f"BERT embedding error: {e}", exc_info=True)
        return np.zeros(768)

def build_cognitive_graph(videos, transcripts):
    try:
        graph = nx.Graph()
        for video, transcript in zip(videos, transcripts):
            video_title = video['title']
            transcript_text = " ".join([seg['text'] for seg in transcript])
            
            graph.add_node(video_title, type="video")
            extract_entities_and_relationships(video['description'], graph)
            extract_entities_and_relationships(transcript_text, graph)
        return graph
    except Exception as e:
        logging.error(f"Graph build error: {e}", exc_info=True)
        return nx.Graph()

def rank_videos(query, videos, transcripts):
    try:
        # Combine title, description and transcript for each video
        video_texts = [
            f"{video['title']} {video['description']} {' '.join(seg['text'] for seg in transcript)}"
            for video, transcript in zip(videos, transcripts)
        ]
        
        query_embed = extract_bert_embeddings(query)
        video_embeds = [extract_bert_embeddings(text) for text in video_texts]
        
        similarities = cosine_similarity([query_embed], video_embeds)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Convert numpy types to native Python types
        return [(videos[i], float(similarities[i])) for i in ranked_indices]
    except Exception as e:
        logging.error(f"Ranking error: {e}", exc_info=True)
        return list(zip(videos, [0.0]*len(videos)))

def visualize_graph(top_ranked_videos, graph):
    try:
        plt.clf()
        subgraph = nx.Graph()
        
        for video, score in top_ranked_videos[:3]:
            video_title = video['title']
            subgraph.add_node(video_title, type='video', score=float(score))
            
            doc = nlp(video['description'])
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            for entity_text, entity_label in entities:
                subgraph.add_node(entity_text, type=entity_label)
                subgraph.add_edge(video_title, entity_text)
        
        pos = nx.spring_layout(subgraph)
        node_colors = ['#FF9999' if subgraph.nodes[n].get('type') == 'video' else '#99FF99' 
                      for n in subgraph.nodes]
        
        plt.figure(figsize=(10, 8))
        nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, 
               node_size=2500, font_size=10, font_weight='bold')
        plt.savefig('static/images/graph.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Visualization error: {e}", exc_info=True)

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def extract_frames(video_path, frame_folder):
    os.makedirs(frame_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        path = os.path.join(frame_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(path, frame)
        frame_count += 1
    video.release()
    return frame_count

def upscale_frames_executable(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    exe_path = os.path.join("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan")

    for file in input_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        command = [
            exe_path,
            "-i", input_path,
            "-o", output_path,
            "-n", "realesrgan-x4plus"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_video_from_frames(frame_folder, output_video, fps=30):
    files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    if not files:
        return False

    frame_path = os.path.join(frame_folder, files[0])
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for file in files:
        img = cv2.imread(os.path.join(frame_folder, file))
        out.write(img)

    out.release()
    return True

def merge_audio(original_video, upscaled_video, final_output):
    command = [
        "ffmpeg", "-y",
        "-i", upscaled_video,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_output
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(final_output)

# Update your summarize function generation parameters
def summarize(text):
    inputs = bart_tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    ).to(device)
    
    summary_ids = bart_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=4,  # Must be >1 when using early_stopping
        early_stopping=True,
        max_length=150
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@app.route('/enhance', methods=['POST'])
def enhance_video():
    video_id = request.form.get('video_id')
    session['enhance_progress'] = 0
    session['enhance_logs'] = ["1/5: Starting video enhancement..."]
    session.pop('progress', None)
    session.pop('logs', None)
    
    try:
        session['enhance_logs'].append("2/5: Downloading original video...")
        session['enhance_progress'] = 20
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        original_path = download_video(video_url)
        
        video_dir = os.path.join('downloads', video_id)
        os.makedirs(video_dir, exist_ok=True)
        frame_folder = os.path.join(video_dir, 'frames')
        upscaled_folder = os.path.join(video_dir, 'upscaled_frames')
        enhanced_path = os.path.join(video_dir, 'enhanced.mp4')
        final_path = os.path.join(video_dir, 'enhanced_with_audio.mp4')
        
        session['enhance_logs'].append("3/5: Extracting frames...")
        session['enhance_progress'] = 40
        total_frames = extract_frames(original_path, frame_folder)
        
        session['enhance_logs'].append("4/5: Upscaling frames...")
        session['enhance_progress'] = 60
        upscale_frames_executable(frame_folder, upscaled_folder)
        
        session['enhance_logs'].append("5/5: Creating enhanced video...")
        session['enhance_progress'] = 80
        fps = get_video_fps(original_path)
        create_success = create_video_from_frames(upscaled_folder, enhanced_path, fps)
        if not create_success:
            raise Exception("Failed to create video from frames")
        
        merge_success = merge_audio(original_path, enhanced_path, final_path)
        if not merge_success:
            raise Exception("Audio merge failed")
        
        session['enhance_progress'] = 100
        session['enhance_logs'].append("✅ Enhancement complete!")
        return jsonify({"success": True, "enhanced_path": final_path})
    
    except Exception as e:
        logging.error(f"Enhancement error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download-enhanced/<video_id>')
def download_enhanced(video_id):
    video_dir = os.path.join('downloads', video_id)
    enhanced_path = os.path.join(video_dir, 'enhanced_with_audio.mp4')
    if os.path.exists(enhanced_path):
        return send_file(enhanced_path, as_attachment=True, download_name=f"enhanced_{video_id}.mp4")
    return "Enhanced video not found", 404

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.clear()
        query = request.form.get('query', '').strip()
        
        try:
            if not query:
                raise ValueError("Empty search query")
                
            results = search_youtube(query)
            if not results:
                raise ValueError("No videos found")
            
            transcripts = []
            for vid in results:
                try:
                    vid_url = f"https://youtube.com/watch?v={vid['video_id']}"
                    file_path = download_video(vid_url)
                    segments = transcribe_video(file_path)
                    transcripts.append(segments)
                except Exception as e:
                    logging.error(f"Video processing failed: {e}", exc_info=True)
                    continue

            ranked_results = rank_videos(query, results, transcripts)
            # Ensure all values are JSON serializable
            session['results'] = [(video, float(score)) for video, score in ranked_results]
            
            cognitive_graph = build_cognitive_graph(results, transcripts)
            visualize_graph(ranked_results, cognitive_graph)
            return redirect(url_for('results'))
            
        except Exception as e:
            logging.error(f"Search failed: {e}", exc_info=True)
            return render_template('index.html', 
                                 error=f"Search failed: {str(e)}")
    
    return render_template('index.html')

@app.route('/quick-search', methods=['POST'])
def quick_search():
    try:
        session.clear()
        query = request.form.get('query', '').strip()
        if not query:
            return redirect(url_for('index'))
        
        results = search_youtube(query)
        if not results:
            return redirect(url_for('index'))
            
        session['quick_results'] = [result['video_id'] for result in results]
        return redirect(url_for('video', vid_id=results[0]['video_id']))
        
    except Exception as e:
        logging.error(f"Quick search failed: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/suggest', methods=['GET'])
def fetch_youtube_suggestions():
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify([])

        response = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={
                "client": "firefox",
                "ds": "yt",
                "q": query,
                "hl": "en"
            },
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            },
            timeout=2
        )
        if response.status_code == 200:
            suggestions = response.json()[1]
            return jsonify(suggestions[:5]), 200, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        return jsonify([])
    except Exception as e:
        logging.error(f"Suggestions error: {e}", exc_info=True)
        return jsonify([])

@app.route('/results')
def results():
    return render_template('results.html', 
                         results=session.get('results', []),
                         graph_url='static/images/graph.png')

# @app.route('/video/<vid_id>')  # This should exist ONLY ONCE in your code
# def video(vid_id):
#     try:
#         video = next((v for v, _ in session.get('results', []) if v['video_id'] == vid_id), None)
#         if not video:
#             return redirect(url_for('index'))
        
#         vid_url = f"https://youtube.com/watch?v={vid_id}"
#         file_path = download_video(vid_url)
#         segments = transcribe_video(file_path)
#         transcript_text = " ".join([seg['text'] for seg in segments])
#         summary = summarize(transcript_text)
        
#         return render_template('video.html', 
#                             video=video,
#                             segments=segments,
#                             current_time=request.args.get('t', 0),
#                             summary=summary)
    
#     except Exception as e:
#         logging.error(f"Video load failed: {e}", exc_info=True)
#         return redirect(url_for('index'))

@app.route('/video/<vid_id>')
def video(vid_id):
    try:
        video = next((v for v, _ in session.get('results', []) if v['video_id'] == vid_id), None)
        if not video:
            return redirect(url_for('index'))
        
        # Get timestamp as integer to prevent YouTube API issues
        current_time = request.args.get('t', default=0, type=int)
        
        vid_url = f"https://youtube.com/watch?v={vid_id}"
        file_path = download_video(vid_url)
        segments = transcribe_video(file_path)
        transcript_text = " ".join([seg['text'] for seg in segments])
        summary = summarize(transcript_text)
        
        return render_template('video.html', 
                            video=video,
                            segments=segments,
                            current_time=current_time,
                            summary=summary)
    
    except Exception as e:
        logging.error(f"Video load failed: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/play_video/<video_id>')
def play_video(video_id):
    try:
        current_time = request.args.get('t', 0, type=float)
        return render_template('video_player.html', 
                             video_id=video_id,
                             current_time=current_time)
    except Exception as e:
        logging.error(f"Play video failed: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/download/<vid_id>')
def download(vid_id):
    try:
        video = next((v for v, _ in session.get('results', []) if v['video_id'] == vid_id), None)
        if not video:
            return redirect(url_for('index'))
        
        segments = transcribe_video(f"downloads/{vid_id}.mp4")
        srt = []
        for i, seg in enumerate(segments):
            srt.append(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n")
        
        return send_file(
            io.BytesIO("".join(srt).encode()),
            mimetype='text/plain',
            download_name=f"{vid_id}_transcript.srt",
            as_attachment=True
        )
    
    except Exception as e:
        logging.error(f"Download failed: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze_video():
    video_id = request.form.get('video_id')
    use_enhanced = request.form.get('enhanced', 'false') == 'true'
    
    session['progress'] = 0
    session['logs'] = ["1/4: Initializing analysis..."]
    
    try:
        video_path = f"downloads/{video_id}/enhanced_with_audio.mp4" if use_enhanced \
                   else download_video(f"https://www.youtube.com/watch?v={video_id}")

        session['logs'].append("3/4: Analyzing video content...")
        session['progress'] = 50
        
        detections = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 10)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % (total_frames // 10) == 0:
                progress = min(50 + int(40 * (frame_count / total_frames)), 90)
                session['progress'] = progress
                session['logs'].append(f"Processed {frame_count}/{total_frames} frames")
            
            if frame_count % frame_interval == 0:
                results = yolo_model(frame)
                timestamp = frame_count / fps
                objects = list(set([
                    yolo_model.names[int(box.cls)] 
                    for box in results[0].boxes
                ]))
                
                if objects:
                    detections.append({
                        "timestamp": int(timestamp),
                        "objects": objects,
                        "embed_url": f"https://www.youtube.com/embed/{video_id}?start={int(timestamp)}&autoplay=1"
                    })
            
            frame_count += 1
        
        session['progress'] = 100
        session['logs'].append("4/4: Analysis complete!")
        cap.release()
        if not use_enhanced:
            os.remove(video_path)
        return jsonify(detections)
    
    except Exception as e:
        if not use_enhanced and 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/progress')
def get_progress():
    return jsonify({
        "progress": session.get('progress', 0),
        "logs": session.get('logs', []),
        "enhance_progress": session.get('enhance_progress', 0),
        "enhance_logs": session.get('enhance_logs', [])
    })

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Server error"), 500

if __name__ == '__main__':
    app.run(debug=False)