# from deepface import DeepFace
# import numpy as np
# import cv2
# from mtcnn import MTCNN

# def cosine_similarity(embedding1, embedding2):
#     """độ tương đồng cosine giữa hai vector embedding"""
#     e1, e2 = np.array(embedding1), np.array(embedding2)
#     return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

# def detect_faces(image):
#     """Nhận diện khuôn mặt trong ảnh"""
#     if image is None:
#         print("⚠ Lỗi: Ảnh đầu vào bị rỗng!")
#         return []
    
#     detector = MTCNN()
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển BGR sang RGB
#     results = detector.detect_faces(rgb_image)

#     faces = []
#     for res in results:
#         x, y, width, height = res['box']
#         x, y = max(0, x), max(0, y)
#         face = image[y:y + height, x:x + width]
#         faces.append(face)

#     return faces

# def get_embeddings(faces, model_name="Facenet512"):
#     """trích xuất embedding từ danh sách khuôn mặt."""
#     embeddings = []
#     for i, face in enumerate(faces):
#         face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         try:
#             embedding_result = DeepFace.represent(img_path=face_rgb, model_name=model_name, enforce_detection=False)
#             if embedding_result:
#                 embeddings.append(embedding_result[0]["embedding"])
#         except Exception as e:
#             print(f"⚠ Lỗi khi trích xuất embedding khuôn mặt {i + 1}: {str(e)}")

#     return embeddings

# def compare_faces(image1_path, image2_path, model_name="Facenet512", threshold=0.6):
#     """so sánh tất cả khuôn mặt giữa hai ảnh"""
#     image1 = cv2.imread(image1_path)
#     image2 = cv2.imread(image2_path)

#     faces1 = detect_faces(image1)
#     faces2 = detect_faces(image2)

#     if not faces1 or not faces2:
#         print("⚠ Không tìm thấy khuôn mặt trong một hoặc cả hai ảnh.")
#         return []

#     embeddings1 = get_embeddings(faces1, model_name)
#     embeddings2 = get_embeddings(faces2, model_name)

#     results = []
#     for i, emb1 in enumerate(embeddings1):
#         for j, emb2 in enumerate(embeddings2):
#             similarity = cosine_similarity(emb1, emb2)
#             is_similar = similarity > threshold
#             results.append({
#                 "face1_index": i + 1,
#                 "face2_index": j + 1,
#                 "similarity": similarity,
#                 "match": is_similar
#             })

#     return sorted(results, key=lambda x: x["similarity"], reverse=True)

# def extract_faces_from_video(video_path, frame_interval=30):
#     """trích xuất khuôn mặt từ video """
#     cap = cv2.VideoCapture(video_path)
#     faces_list = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_interval == 0:
#             faces = detect_faces(frame)
#             faces_list.extend(faces)

#         frame_count += 1

#     cap.release()
#     return faces_list

# def compare_faces_in_video(video_path, reference_image_path, model_name="Facenet512", threshold=0.5):
#     """so sánh tất cả khuôn mặt trong video với khuôn mặt sánh khuôn mặt trong trong ảnh."""
#     ref_image = cv2.imread(reference_image_path)
#     ref_faces = detect_faces(ref_image)

#     if not ref_faces:
#         print("⚠ Không tìm thấy khuôn mặt trong ảnh tham chiếu.")
#         return []

#     ref_embeddings = get_embeddings(ref_faces, model_name)
#     video_faces = extract_faces_from_video(video_path)

#     if not video_faces:
#         print("⚠ Không tìm thấy khuôn mặt trong video.")
#         return []

#     video_embeddings = get_embeddings(video_faces, model_name)
#     results = []

#     for i, ref_emb in enumerate(ref_embeddings):
#         for j, vid_emb in enumerate(video_embeddings):
#             similarity = cosine_similarity(ref_emb, vid_emb)
#             is_similar = similarity > threshold
#             results.append({
#                 "ref_face_index": i + 1,
#                 "video_face_index": j + 1,
#                 "similarity": similarity,
#                 "match": is_similar
#             })

#     return sorted(results, key=lambda x: x["similarity"], reverse=True)
# def compare_faces_between_videos(video1_path, video2_path, model_name="Facenet512", threshold=0.5):
#     """so sánh tất cả khuôn mặt giữa hai video."""
#     faces_video1 = extract_faces_from_video(video1_path)
#     faces_video2 = extract_faces_from_video(video2_path)

#     if not faces_video1 or not faces_video2:
#         print("⚠ Không tìm thấy khuôn mặt trong một hoặc cả hai video.")
#         return []

#     embeddings_video1 = get_embeddings(faces_video1, model_name)
#     embeddings_video2 = get_embeddings(faces_video2, model_name)

#     results = []
#     for i, emb1 in enumerate(embeddings_video1):
#         for j, emb2 in enumerate(embeddings_video2):
#             similarity = cosine_similarity(emb1, emb2)
#             is_similar = similarity > threshold
#             results.append({
#                 "video1_face_index": i + 1,
#                 "video2_face_index": j + 1,
#                 "similarity": similarity,
#                 "match": is_similar
#             })
#     return sorted(results, key=lambda x: x["similarity"], reverse=True)
