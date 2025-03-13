import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os

# Definir el modelo OCR con CNN + LSTM
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # La salida de la CNN tiene un tamaño [batch, 256, 4, 16], por lo que el input_size debe ser 256 * 4 = 1024
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)  # bidireccional LSTM → 2 * hidden_size

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        batch_size, channels, height, width = x.shape  # [batch, 256, 4, 16]

        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, width, channels, height]
        x = x.view(batch_size, width, channels * height)  # [1, 16, 1024] → ahora coincide con input_size de LSTM

        x, _ = self.lstm(x)
        x = self.fc(x)  # [batch, width, num_classes]
        return x

# Función de entrenamiento con guardado de checkpoint
def train_ocr(image_path, expected_text, epochs=100, checkpoint_path="ocr_checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones para la imagen
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("L")  # Convertir a escala de grises
    image_tensor = transform(image).unsqueeze(0).to(device)  # Agregar dimensión de batch

    # Definir modelo y optimizador
    num_classes = 100  # Número de clases en OCR (ajustar según el dataset)
    model = OCRModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0

    # Cargar checkpoint si existe
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        print(f"🔄 Checkpoint encontrado. Reanudando desde la epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()
        output = model(image_tensor)  # Forward pass
        loss = criterion(output.view(-1, num_classes), torch.randint(0, num_classes, (output.size(0) * output.size(1),)).to(device))  
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Guardar checkpoint cada 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, checkpoint_path)
            print(f"✅ Checkpoint guardado en epoch {epoch+1}")

# Datos de entrenamiento
expected_text = """PRUEBAS HÚMEDO
Solidez del Color a:
Lavado
A: 4
B: 4
C: 3
Piscina
B: 4
Mar
A: 3
B: 5
C: 4.5
ESTABILIDAD DIMENSIONAL
Encogimiento %:
Ancho: -2.6
Largo: -7.6
PRUEBAS FÍSICAS
Peso (g/m²): 263
Mallas (cm): 63
Columnas (pulg): 54
Raport (cm): 60
Elongación/Recuperac. Ancho (%): 10/20
Elongación/Recuperac. Largo (%): 50/69
Entorche Ancho: B
Entorche Largo: B"""

# Ruta de la imagen
image_path = "formato1.jpeg"

# Ejecutar entrenamiento con checkpoint
train_ocr(image_path, expected_text, epochs=5000)
