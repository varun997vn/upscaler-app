package com.example.upscaler

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.system.measureTimeMillis

class Upscaler(
    context: Context,
    modelFileName: String,
    private val useNpu: Boolean = false,
) {
    private val interpreter: Interpreter
    private var nnApiDelegate: NnApiDelegate? = null

    val inputWidth: Int
    val inputHeight: Int
    val outputWidth: Int
    val outputHeight: Int
    val modelScale: Int

    init {
        val modelBuffer = loadModelFile(context, modelFileName)
        val options = Interpreter.Options()
        if (useNpu) {
            nnApiDelegate = NnApiDelegate()
            options.addDelegate(nnApiDelegate)
        }
        interpreter = Interpreter(modelBuffer, options)

        val inputShape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]

        val outputShape = interpreter.getOutputTensor(0).shape() // [1, H, W, 3]
        outputHeight = outputShape[1]
        outputWidth = outputShape[2]
        modelScale = outputWidth / inputWidth
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
     */
    fun runInference(input: FloatArray, output: FloatArray): Long {
        val inputBuffer = ByteBuffer.allocateDirect(input.size * 4).apply {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().put(input)
        }
        val outputBuffer = ByteBuffer.allocateDirect(output.size * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val elapsedMs = measureTimeMillis {
            interpreter.run(inputBuffer, outputBuffer)
        }

        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(output)
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
        nnApiDelegate?.close()
        nnApiDelegate = null
    }
}

data class UpscaleResult(val bitmap: Bitmap, val inferenceTimeMs: Long)
