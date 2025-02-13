from deepface import DeepFace
import numpy as np
import cv2
from mtcnn import MTCNN
import os
from sklearn.cluster import DBSCAN

def cosine_similarity(embedding1, embedding2):
    '''Tính độ tương đồng cosine giữa hai vector embedding'''
    e1, e2 = np.array(embedding1), np.array(embedding2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def detect_faces(image):
    '''Nhận diện khuôn mặt trong ảnh'''
    if image is None:
        print("⚠ Lỗi: Ảnh đầu vào bị rỗng!")
        return []
    
    detector = MTCNN()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    results = detector.detect_faces(rgb_image)

    faces = []
    for res in results:
        x, y, width, height = res['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y + height, x:x + width]
        faces.append(face)

    return faces

def get_embeddings(faces, model_name="Facenet512"):
    '''Trích xuất embedding từ danh sách khuôn mặt'''
    embeddings = []
    for i, face in enumerate(faces):
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            embedding_result = DeepFace.represent(img_path=face_rgb, model_name=model_name, enforce_detection=False)
            if embedding_result:
                embeddings.append(list(embedding_result[0]["embedding"]))
        except Exception as e:
            print(f"⚠ Lỗi khi trích xuất embedding khuôn mặt {i + 1}: {str(e)}")

    return embeddings

def save_detected_faces(faces, base_filename, folder="static/faces"):
    '''Lưu ảnh khuôn mặt đã nhận diện'''
    os.makedirs(folder, exist_ok=True)
    saved_faces = []
    
    for i, face in enumerate(faces):
        face_filename = f"{folder}/{base_filename}_face_{i}.jpg"
        cv2.imwrite(face_filename, face)
        saved_faces.append(face_filename)
    
    return saved_faces

def compare_faces(image1_path, image2_path, faces_folder):
    '''So sánh khuôn mặt giữa 2 ảnh'''
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    faces1 = detect_faces(image1)
    faces2 = detect_faces(image2)

    faces1_saved = save_detected_faces(faces1, "image1", faces_folder)
    faces2_saved = save_detected_faces(faces2, "image2", faces_folder)

    embeddings1 = get_embeddings(faces1)
    embeddings2 = get_embeddings(faces2)

    results = []
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            similarity = cosine_similarity(emb1, emb2)
            match = similarity > 0.6
            results.append({
                "face1_index": i + 1,
                "face2_index": j + 1,
                "similarity": similarity,
                "match": match,
                "face1_img": faces1_saved[i],
                "face2_img": faces2_saved[j]
            })
    
    return results, faces1_saved, faces2_saved

def extract_faces_from_video(video_path, frame_interval=30):
    '''Trích xuất khuôn mặt từ video'''
    cap = cv2.VideoCapture(video_path)
    faces_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            faces = detect_faces(frame)
            faces_list.extend(faces)

        frame_count += 1

    cap.release()
    return faces_list

def compare_faces_in_video(video_path, image_path, faces_folder):
    '''So sánh khuôn mặt giữa ảnh và video'''
    image = cv2.imread(image_path)
    faces_img = detect_faces(image)
    faces_img_saved = save_detected_faces(faces_img, "image_target", faces_folder)
    
    video_faces = extract_faces_from_video(video_path)
    faces_vid_saved = save_detected_faces(video_faces, "video", faces_folder)
    embeddings_img = get_embeddings(faces_img)
    embeddings_vid = get_embeddings(video_faces)
    
    clustered_faces_vid, labels_vid = cluster_faces(embeddings_vid, faces_vid_saved)

    results = []
    for label, faces_group in clustered_faces_vid.items():
        representative_face = get_representative_face(embeddings_vid, faces_vid_saved, labels_vid, label)
        if representative_face:
            for i, emb1 in enumerate(embeddings_img):
                representative_face_embedding = DeepFace.represent(img_path=representative_face, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                similarity = cosine_similarity(emb1, representative_face_embedding)
                match = similarity > 0.6
                results.append({
                    "face1_index": i + 1,
                    "face2_index": label + 1,
                    "similarity": similarity,
                    "match": match,
                    "face1_img": faces_img_saved[i],
                    "face2_img": representative_face
                })

    return results, faces_img_saved, faces_vid_saved

def compare_faces_between_videos(video1_path, video2_path, faces_folder):
    '''So sánh khuôn mặt giữa 2 video'''
    faces_vid1 = extract_faces_from_video(video1_path)
    faces_vid2 = extract_faces_from_video(video2_path)

    faces_vid1_saved = save_detected_faces(faces_vid1, "video1", faces_folder)
    faces_vid2_saved = save_detected_faces(faces_vid2, "video2", faces_folder)

    embeddings_vid1 = get_embeddings(faces_vid1)
    embeddings_vid2 = get_embeddings(faces_vid2)

    clustered_faces_vid1, labels_vid1 = cluster_faces(embeddings_vid1, faces_vid1_saved)
    clustered_faces_vid2, labels_vid2 = cluster_faces(embeddings_vid2, faces_vid2_saved)

    results = []

    for label1, faces_group1 in clustered_faces_vid1.items():
        representative_face_vid1 = get_representative_face(embeddings_vid1, faces_vid1_saved, labels_vid1, label1)
        if representative_face_vid1:
            for label2, faces_group2 in clustered_faces_vid2.items():
                representative_face_vid2 = get_representative_face(embeddings_vid2, faces_vid2_saved, labels_vid2, label2)
                if representative_face_vid2:
                    representative_face_embedding_vid1 = DeepFace.represent(img_path=representative_face_vid1, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                    representative_face_embedding_vid2 = DeepFace.represent(img_path=representative_face_vid2, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                    
                    similarity = cosine_similarity(representative_face_embedding_vid1, representative_face_embedding_vid2)
                    match = similarity > 0.6
                    results.append({
                        "face_index": label1 + 1,
                        "face2_index": label2 + 1,
                        "similarity": similarity,
                        "match": match,
                        "face1_img": representative_face_vid1,
                        "face2_img": representative_face_vid2
                    })

    return results, faces_vid1_saved, faces_vid2_saved

def get_representative_face(embeddings, faces_saved, labels, label):
    '''Lấy khuôn mặt đại diện của mỗi nhóm khuôn mặt'''
    group_embeddings = [embeddings[i] for i in range(len(embeddings)) if labels[i] == label]
    if not group_embeddings:
        return None
    representative_embedding = np.mean(group_embeddings, axis=0)
    min_distance = float('inf')
    representative_face = None
    for i, emb in enumerate(embeddings):
        distance = np.linalg.norm(emb - representative_embedding)
        if distance < min_distance:
            min_distance = distance
            representative_face = faces_saved[i]
    
    return representative_face

def cluster_faces(embeddings, faces_saved, eps=0.6, min_samples=2):
    '''Phân cụm các khuôn mặt dựa trên embedding'''
    if len(embeddings) == 0:
        print("⚠ Không có embedding nào để phân cụm!")
        return {}, []

    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_

    clustered_faces = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  
        if label not in clustered_faces:
            clustered_faces[label] = []
        clustered_faces[label].append(faces_saved[idx])
    
    return clustered_faces, labels
