// Arduino Nano 33 BLE Sense — Anomaly Detection Sketch
// Dissertation: TinyML-Based Anomaly Detection
// Author: Thondupu Dileep | 2024AB05233

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include "anomaly_detection_model.h"   // our exported C array

#define WINDOW_SIZE    2048
#define SAMPLE_RATE    48000
#define THRESHOLD      0.000312f   // p95 from training
#define TENSOR_ARENA   8192        // 8 KB arena

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
uint8_t tensor_arena[TENSOR_ARENA];
float signal_buffer[WINDOW_SIZE];

void setup() {
    Serial.begin(115200);

    model       = tflite::GetModel(anomaly_detection_model);
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA);
    interpreter->AllocateTensors();

    Serial.println("Anomaly detector ready");
}

void loop() {
    // 1. Read 2048 vibration samples from sensor
    collect_sensor_data(signal_buffer, WINDOW_SIZE);

    // 2. Normalise using stored mean/scale from training
    normalise(signal_buffer, WINDOW_SIZE);

    // 3. Copy into model input tensor
    float* input = interpreter->input(0)->data.f;
    memcpy(input, signal_buffer, WINDOW_SIZE * sizeof(float));

    // 4. Run inference
    interpreter->Invoke();

    // 5. Compute reconstruction MSE
    float* output = interpreter->output(0)->data.f;
    float mse = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float diff = input[i] - output[i];
        mse += diff * diff;
    }
    mse /= WINDOW_SIZE;

    // 6. Anomaly decision
    if (mse > THRESHOLD) {
        Serial.print("ANOMALY DETECTED — MSE: ");
        Serial.println(mse, 6);
        digitalWrite(LED_RED, HIGH);   // alert LED
    } else {
        Serial.print("Normal — MSE: ");
        Serial.println(mse, 6);
        digitalWrite(LED_GREEN, HIGH);
    }

    delay(100);  // slide window every 100ms
}
