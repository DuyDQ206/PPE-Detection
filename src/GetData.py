from roboflow import Roboflow
rf = Roboflow(api_key="ILJPqhY8441Ca5WZ5YMT")
project = rf.workspace("constructionsite").project("ppe-fruwx")
version = project.version(5)
dataset = version.download(target_format = "yolov8",
                            location = "data/")