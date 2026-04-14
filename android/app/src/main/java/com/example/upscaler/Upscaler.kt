package com.example.upscaler

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.system.measureTimeMillis

private const val TAG = "Upscaler"

enum class Accelerator { CPU, GPU }

class Upscaler(
    context: Context,
    modelFileName: String,
    private val accelerator: Accelerator = Accelerator.CPU,
) {
    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    val inputWidth: Int
    val inputHeight: Int
    val outputWidth: Int
    val outputHeight: Int
    val modelScale: Int
    private val inputDataType: DataType
    private val outputDataType: DataType

    /** Human-readable status of the chosen accelerator (visible in UI). */
    val acceleratorStatus: String

    init {
        val modelBuffer = loadModelFile(context, modelFileName)
        val options = Interpreter.Options()
        options.numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)

        acceleratorStatus = when (accelerator) {
            Accelerator.CPU -> {
                Log.i(TAG, "Using CPU (XNNPACK), threads=${options.numThreads}")
                "CPU (XNNPACK, ${options.numThreads}t)"
            }

            Accelerator.GPU -> {
                val compat = CompatibilityList()
                if (!compat.isDelegateSupportedOnThisDevice) {
                    Log.w(TAG, "GPU delegate NOT supported on this device — falling back to CPU")
                    "GPU NOT SUPPORTED (CPU fallback)"
                } else {
                    try {
                        val gpuOpts: GpuDelegateFactory.Options = compat.bestOptionsForThisDevice
                        gpuDelegate = GpuDelegate(gpuOpts)
                        options.addDelegate(gpuDelegate)
                        Log.i(TAG, "GPU delegate attached (OpenCL/OpenGL)")
                        "GPU delegate attached"
                    } catch (e: Throwable) {
                        Log.e(TAG, "GPU delegate init FAILED — CPU fallback: ${e.message}", e)
                        gpuDelegate?.close(); gpuDelegate = null
                        "GPU FAILED (CPU): ${e.message}"
                    }
                }
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        val inputShape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        inputDataType = interpreter.getInputTensor(0).dataType()

        val outputShape = interpreter.getOutputTensor(0).shape() // [1, H, W, 3]
        outputHeight = outputShape[1]
        outputWidth = outputShape[2]
        outputDataType = interpreter.getOutputTensor(0).dataType()
        modelScale = outputWidth / inputWidth

        Log.i(TAG, "Upscaler ready — accelerator=$accelerator status=$acceleratorStatus " +
            "input=${inputWidth}x${inputHeight} output=${outputWidth}x${outputHeight} " +
            "dtype=$inputDataType model=$modelFileName")
    }

    private fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
        context.assets.openFd(fileName).use { afd ->
            FileInputStream(afd.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    afd.startOffset,
                    afd.declaredLength,
                )
            }
        }
    }

    /**
     * Runs the model on the given input and returns the execution time in milliseconds.
     * Output is written into [output], which must be sized to match the model's output tensor.
     * Handles both FLOAT32 and UINT8/INT8 quantized models transparently.
     */
    fun runInference(input: FloatArray, output: FloatArray): Long {
        val bytesPerInputElement = if (inputDataType == DataType.FLOAT32) 4 else 1
        val bytesPerOutputElement = if (outputDataType == DataType.FLOAT32) 4 else 1

        val inputBuffer = ByteBuffer.allocateDirect(input.size * bytesPerInputElement).apply {
            order(ByteOrder.nativeOrder())
            when (inputDataType) {
                DataType.FLOAT32 -> asFloatBuffer().put(input)
                DataType.UINT8 -> input.forEach { put(it.toInt().coerceIn(0, 255).toByte()) }
                DataType.INT8 -> input.forEach { put((it.toInt().coerceIn(0, 255) - 128).toByte()) }
                else -> throw IllegalStateException("Unsupported input data type: $inputDataType")
            }
        }
        val outputBuffer = ByteBuffer.allocateDirect(output.size * bytesPerOutputElement).apply {
            order(ByteOrder.nativeOrder())
        }

        val elapsedMs = measureTimeMillis {
            interpreter.run(inputBuffer, outputBuffer)
        }
        Log.d(TAG, "Inference ${elapsedMs}ms — delegate=$acceleratorStatus")

        outputBuffer.rewind()
        when (outputDataType) {
            DataType.FLOAT32 -> outputBuffer.asFloatBuffer().get(output)
            DataType.UINT8 -> output.indices.forEach { output[it] = (outputBuffer.get().toInt() and 0xFF).toFloat() }
            DataType.INT8 -> output.indices.forEach { output[it] = ((outputBuffer.get().toInt() and 0xFF)).toFloat() }
            else -> throw IllegalStateException("Unsupported output data type: $outputDataType")
        }
        return elapsedMs
    }

    /**
     * Upscales [bitmap] by [scale]. The model runs in 50×50 → 200×200 tiles (4× native);
     * the tiled 4× result is then resampled to match the requested scale factor.
     */
    fun upscale(bitmap: Bitmap, scale: Float = modelScale.toFloat()): UpscaleResult {
        val origW = bitmap.width
        val origH = bitmap.height
        val tilesX = ((origW + inputWidth - 1) / inputWidth).coerceAtLeast(1)
        val tilesY = ((origH + inputHeight - 1) / inputHeight).coerceAtLeast(1)
        val paddedW = tilesX * inputWidth
        val paddedH = tilesY * inputHeight

        val padded = if (paddedW == origW && paddedH == origH) bitmap
        else Bitmap.createScaledBitmap(bitmap, paddedW, paddedH, true)

        val fullW = paddedW * modelScale
        val fullH = paddedH * modelScale
        val fullBitmap = Bitmap.createBitmap(fullW, fullH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(fullBitmap)

        val output = FloatArray(outputWidth * outputHeight * 3)
        var totalMs = 0L
        for (ty in 0 until tilesY) {
            for (tx in 0 until tilesX) {
                val tile = Bitmap.createBitmap(
                    padded, tx * inputWidth, ty * inputHeight, inputWidth, inputHeight,
                )
                val input = bitmapToFloatArray(tile)
                totalMs += runInference(input, output)
                val outTile = floatArrayToBitmap(output, outputWidth, outputHeight)
                canvas.drawBitmap(
                    outTile,
                    (tx * outputWidth).toFloat(),
                    (ty * outputHeight).toFloat(),
                    null,
                )
            }
        }

        val targetW = (origW * scale).toInt().coerceAtLeast(1)
        val targetH = (origH * scale).toInt().coerceAtLeast(1)
        val result = if (targetW == fullW && targetH == fullH) fullBitmap
        else Bitmap.createScaledBitmap(fullBitmap, targetW, targetH, true)
        return UpscaleResult(result, totalMs)
    }

    private fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val floats = FloatArray(pixels.size * 3)
        for (i in pixels.indices) {
            val p = pixels[i]
            floats[i * 3] = Color.red(p).toFloat()
            floats[i * 3 + 1] = Color.green(p).toFloat()
            floats[i * 3 + 2] = Color.blue(p).toFloat()
        }
        return floats
    }

    private fun floatArrayToBitmap(data: FloatArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = data[i * 3].coerceIn(0f, 255f).toInt()
            val g = data[i * 3 + 1].coerceIn(0f, 255f).toInt()
            val b = data[i * 3 + 2].coerceIn(0f, 255f).toInt()
            pixels[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close(); gpuDelegate = null
    }
}

data class UpscaleResult(val bitmap: Bitmap, val inferenceTimeMs: Long)
