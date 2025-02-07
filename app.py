from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from compare import compare_faces, compare_faces_in_video, compare_faces_between_videos
# from compare_face import compare_faces, compare_faces_in_video, compare_faces_between_videos

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'static/faces'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER

# Tạo thư mục nếu chưa có
for folder in [UPLOAD_FOLDER, FACES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        compare_type = request.form.get("compare_type")

        if compare_type == "image_image":
            img1 = request.files["image1"]
            img2 = request.files["image2"]
            if img1 and img2:
                path1 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(img1.filename))
                path2 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(img2.filename))
                img1.save(path1)
                img2.save(path2)

                results, faces1, faces2 = compare_faces(path1, path2, app.config["FACES_FOLDER"])

                return render_template("results.html", results=results, 
                                       file1=img1.filename, file2=img2.filename,
                                       detected_faces1=faces1, detected_faces2=faces2)

        elif compare_type == "image_video":
            img = request.files["image"]
            vid = request.files["video"]
            if img and vid:
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(img.filename))
                vid_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(vid.filename))
                img.save(img_path)
                vid.save(vid_path)

                results, faces_img, faces_vid = compare_faces_in_video(vid_path, img_path, app.config["FACES_FOLDER"])

                return render_template("results.html", results=results, 
                                       file1=img.filename, file2=vid.filename,
                                       detected_faces1=faces_img, detected_faces2=faces_vid)

        elif compare_type == "video_video":
            vid1 = request.files["video1"]
            vid2 = request.files["video2"]
            if vid1 and vid2:
                path1 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(vid1.filename))
                path2 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(vid2.filename))
                vid1.save(path1)
                vid2.save(path2)

                results, faces_vid1, faces_vid2 = compare_faces_between_videos(path1, path2, app.config["FACES_FOLDER"])

                return render_template("results.html", results=results, 
                                       file1=vid1.filename, file2=vid2.filename,
                                       detected_faces1=faces_vid1, detected_faces2=faces_vid2)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
