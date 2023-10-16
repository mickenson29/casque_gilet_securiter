from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
from ultralytics import YOLO
import cv2

# Mettre en cache le modèle YOLO
def load_yolo_model():
    model = YOLO("yolov8n.pt")
    return model

# Charger le modèle
model = load_yolo_model()

# Créer une classe de traitement vidéo
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Faire une prédiction sur le cadre
        results = model(frame)

        # Vérifier si l'objet résultats est une liste
        if type(results) is list:

            # Vérifier s'il existe des objets détectés
            if len(results) > 0:

                # Vérifier si un casque et un gilet de sécurité ont été détectés
                helmet_detected = False
                vest_detected = False
                for result in results:
                    names = result.names
                    if "helmet" in names:
                        helmet_detected = True
                    if "vest" in names:
                        vest_detected = True

                # Alerter l'utilisateur si quelqu'un ne porte pas de casque ou de gilet de sécurité
                if not helmet_detected:
                    print("Casque non détecté")
                if not vest_detected:
                    print("Gilet de sécurité non conforme")

        # Dessiner les carrés de détection
        for result in results:
            x, y, w, h = result.bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Vérifier si la personne porte une veste de sécurité
        for result in results:
            if "vest" in result.names:
                # La personne porte une veste de sécurité
                is_wearing_vest = True
                break
            else:
                # La personne ne porte pas de veste de sécurité
                is_wearing_vest = False

        # Afficher un message indiquant si la personne porte ou ne porte pas de veste de sécurité
        if is_wearing_vest:
            print("La personne porte une veste de sécurité")
        else:
            print("La personne ne porte pas de veste de sécurité")

        return frame

# Configurer les paramètres client
client_settings = ClientSettings(
    request_timeout=60,  # seconds
)

# Lancer le flux vidéo en utilisant webrtc_streamer
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    client_settings=client_settings
)

