from flask import Flask, request, render_template, send_from_directory
from enum import Enum
from pyngrok import ngrok
import logging
import click
import os


from ..dataset import Photo


class MatchType(Enum):
    WRONG_ID = 0
    RIGHT_ID = 1
    AMBIG_ID = -1


class Labeler:
    """
    This runs a Flask app for labeling faces.
    """

    def __init__(
        self,
        names_to_faces,
        photo_id_to_url,
        database,
        port=5000,
        use_ngrok=False,
        # TODO - this can become a package constant in the future, but
        # for now show every face on the labeler.
        max_per_page=100,
        face_to_resolved_faces=None,
    ):
        self.names_to_faces = names_to_faces
        self.face_to_labels = {}
        self.photo_id_to_url = {}
        for photo_id, url in photo_id_to_url.items():
            ## Photo always needs to be hosted
            # if Photo._needs_to_be_hosted(url):
            url = "/uploads/" + f"{photo_id:06}"
            self.photo_id_to_url[photo_id] = url

        self.max_per_page = max_per_page
        self.database = database

        # If this is not None, then each face that is labeled
        # will also result in us labeling all other resolved
        # faces in its stack.
        self.face_to_resolved_faces = face_to_resolved_faces

        __folder__ = os.path.abspath(os.path.dirname(__file__))

        app = Flask(
            __name__,
            template_folder=os.path.join(__folder__, "templates"),
        )

        # This removes the default Flask output to streamline the UI.
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        def secho(text, file=None, nl=None, err=None, color=None, **styles):
            pass

        def echo(text, file=None, nl=None, err=None, color=None, **styles):
            pass

        click.echo = echo
        click.secho = secho

        if use_ngrok:
            http_tunnel = ngrok.connect(database.port)
            print("Connect to the labeler via", http_tunnel.public_url)
        else:
            print("Connect to the labeler via", f"http://127.0.0.1:{port}/")

        def gen_home_html():
            return render_template(
                "labeler_home.html",
                names_and_face_counts=[
                    (name, len(self.names_to_faces[name]))
                    for name in self.names_to_faces
                ],
            )

        @app.route("/")
        def home():
            return gen_home_html()

        @app.route("/exit")
        def exit():
            if use_ngrok:
                ngrok.disconnect(http_tunnel.public_url)
                ngrok.kill()
            # http://web.archive.org/web/20190706125149/http://flask.pocoo.org/snippets/67
            request.environ.get("werkzeug.server.shutdown")()
            return "See you next time!"

        @app.route("/uploads/<path:name>")
        def download_file(name):
            assert all(c.isdigit() for c in name)
            file = name + ".jpg"
            return send_from_directory(
                os.getcwd() + os.sep + self.database.get_photo_dir(),
                file,
                as_attachment=False,
            )

        @app.route("/orig/<path:name>", methods=["GET", "POST"])
        def cropped_image_label(name):
            return label("labeler.html", name)

        @app.route("/crop/<path:name>", methods=["GET", "POST"])
        def original_image_label(name):
            return label("cropped_labeler.html", name)

        def label(template, name):
            if name == "favicon.ico":
                # Unless we return a favicon here
                # it will have a generic error.
                return None

            faces = self.names_to_faces[name]
            num_faces = len(faces)

            if request.method == "GET":
                if num_faces > self.max_per_page:
                    num_faces = self.max_per_page
                    faces = faces[-num_faces:]
                face_ids = [face.face_id for face in faces]
                photo_ids = [face.photo_id for face in faces]
                images = [self.photo_id_to_url[id] for id in photo_ids]
                bbox = [face.bounding_box.get_top_left_width_height() for face in faces]
                id_url_bbox = list(enumerate(zip(face_ids, images, bbox)))
                return render_template(
                    template, name=name, id_url_bbox=id_url_bbox, num_faces=num_faces
                )

            elif request.method == "POST":
                # Remove selected faces from names_to_faces
                if num_faces > self.max_per_page:
                    num_faces = self.max_per_page
                    self.names_to_faces[name] = self.names_to_faces[name][:-num_faces]
                    faces = faces[-num_faces:]
                else:
                    del self.names_to_faces[name]

                # Add annotations to face object via .annotate call
                for (face_id_str, annotation_str) in request.form.to_dict().items():
                    if "annotation_face" not in face_id_str:
                        continue
                    face_id = int(face_id_str.split("_")[-1])
                    annotation_value = int(annotation_str)
                    annotation = MatchType(annotation_value)

                    if face_to_resolved_faces is None:
                        found_match = False
                        for face in faces:
                            if face.face_id == face_id:
                                found_match = True
                                face.annotate(annotation)
                                self.database._save_face_annotation(face)
                                break
                        assert found_match
                    else:
                        assert face_id in face_to_resolved_faces
                        for face in face_to_resolved_faces[face_id].faces:
                            face.annotate(annotation)
                            self.database._save_face_annotation(face)
                return gen_home_html()

        app.run(port=port)

        if use_ngrok:
            ngrok.disconnect(http_tunnel.public_url)
            ngrok.kill()
