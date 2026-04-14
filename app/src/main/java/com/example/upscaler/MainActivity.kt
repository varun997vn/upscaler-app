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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.upscaler.ui.theme.UpscalerTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val MODEL_FILE = "esrgan.tflite"

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

    val upscaler = remember { Upscaler(context, MODEL_FILE, useNpu = false) }
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var upscaledBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var status by remember { mutableStateOf("Pick a low-res image to upscale.") }
    var running by remember { mutableStateOf(false) }
    var scale by remember { mutableStateOf(4) }

    val pickImage = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        if (uri != null) {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                selectedBitmap = BitmapFactory.decodeStream(stream)
                upscaledBitmap = null
                status = "Image loaded: ${selectedBitmap?.width}x${selectedBitmap?.height}. " +
                    "Model input: ${upscaler.inputWidth}x${upscaler.inputHeight}."
            }
        }
    }

    LaunchedEffect(Unit) {
        status = "Ready. Model tile: ${upscaler.inputWidth}x${upscaler.inputHeight} → " +
            "${upscaler.outputWidth}x${upscaler.outputHeight} (native ${upscaler.modelScale}x)."
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(text = status)

        Button(
            onClick = { pickImage.launch("image/*") },
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Pick image") }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Scale:")
            listOf(2, 3, 4).forEach { value ->
                Row(
                    modifier = Modifier.selectable(
                        selected = scale == value,
                        enabled = !running,
                        onClick = { scale = value },
                    ),
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                ) {
                    RadioButton(selected = scale == value, onClick = null)
                    Text("${value}x")
                }
            }
        }

        Button(
            onClick = {
                val bmp = selectedBitmap ?: return@Button
                val chosenScale = scale
                running = true
                status = "Running inference at ${chosenScale}x..."
                scope.launch {
                    val result = withContext(Dispatchers.Default) {
                        upscaler.upscale(bmp, chosenScale.toFloat())
                    }
                    upscaledBitmap = result.bitmap
                    status = "Inference took ${result.inferenceTimeMs} ms. " +
                        "Output: ${result.bitmap.width}x${result.bitmap.height} (${chosenScale}x)."
                    running = false
                }
            },
            enabled = selectedBitmap != null && !running,
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
