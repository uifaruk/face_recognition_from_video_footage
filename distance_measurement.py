import face_recognition


known_a_image = face_recognition.load_image_file("a.jpg")
known_b_image = face_recognition.load_image_file("b.jpg")

obama_face_encoding = face_recognition.face_encodings(known_a_image)[0]
biden_face_encoding = face_recognition.face_encodings(known_b_image)[0]

known_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

image_to_test = face_recognition.load_image_file("abc.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()
