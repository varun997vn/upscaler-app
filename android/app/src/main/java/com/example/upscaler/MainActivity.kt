package com.example.upscaler

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.upscaler.ui.theme.UpscalerTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val MODELS = listOf(
    "esrgan.tflite" to "Standard (FP32)",
    "esrgan_int8.tflite" to "Quantized (INT8)",
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            UpscalerTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    UpscalerScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun UpscalerScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedModel by remember { mutableStateOf(MODELS[0].first) }
    var useNpu by remember { mutableStateOf(false) }
    var upscaler by remember { mutableStateOf<Upscaler?>(null) }
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var upscaledBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var status by remember { mutableStateOf("Loading model...") }
    var running by remember { mutableStateOf(false) }
    var scale by remember { mutableStateOf(4) }

    // Recreate the Upscaler whenever the selected model or NPU setting changes; close the previous one.
    LaunchedEffect(selectedModel, useNpu) {
        val old = upscaler
        status = "Loading model..."
        upscaledBitmap = null
        val new = withContext(Dispatchers.IO) { Upscaler(context, selectedModel, useNpu = useNpu) }
        upscaler = new
        old?.close()
        status = "Ready. Model tile: ${new.inputWidth}x${new.inputHeight} → " +
            "${new.outputWidth}x${new.outputHeight} (native ${new.modelScale}x)."
    }

    val pickImage = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        if (uri != null) {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                selectedBitmap = BitmapFactory.decodeStream(stream)
                upscaledBitmap = null
                val u = upscaler
                status = if (u != null) {
                    "Image loaded: ${selectedBitmap?.width}x${selectedBitmap?.height}. " +
                        "Model input: ${u.inputWidth}x${u.inputHeight}."
                } else {
                    "Image loaded: ${selectedBitmap?.width}x${selectedBitmap?.height}."
                }
            }
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(text = status)

        Text("Model:")
        MODELS.forEach { (fileName, displayName) ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(
                        selected = selectedModel == fileName,
                        enabled = !running,
                        onClick = { selectedModel = fileName },
                    ),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(selected = selectedModel == fileName, onClick = null)
                Text(displayName)
            }
        }

        Button(
            onClick = { pickImage.launch("image/*") },
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Pick image") }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Scale:")
            listOf(2, 3, 4).forEach { value ->
                Row(
                    modifier = Modifier.selectable(
                        selected = scale == value,
                        enabled = !running,
                        onClick = { scale = value },
                    ),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = scale == value, onClick = null)
                    Text("${value}x")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Use NPU (NNAPI / Hexagon)")
            Switch(
                checked = useNpu,
                onCheckedChange = { useNpu = it },
                enabled = !running,
            )
        }

        Button(
            onClick = {
                val bmp = selectedBitmap ?: return@Button
                val u = upscaler ?: return@Button
                val chosenScale = scale
                running = true
                status = "Running inference at ${chosenScale}x..."
                scope.launch {
                    val result = withContext(Dispatchers.Default) {
                        u.upscale(bmp, chosenScale.toFloat())
                    }
                    upscaledBitmap = result.bitmap
                    status = "Inference took ${result.inferenceTimeMs} ms. " +
                        "Output: ${result.bitmap.width}x${result.bitmap.height} (${chosenScale}x)."
                    running = false
                }
            },
            enabled = selectedBitmap != null && upscaler != null && !running,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Upscale") }

        selectedBitmap?.let {
            Text("Input")
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "input",
                modifier = Modifier.fillMaxWidth(),
            )
        }
        upscaledBitmap?.let {
            Text("Upscaled")
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "upscaled",
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}
