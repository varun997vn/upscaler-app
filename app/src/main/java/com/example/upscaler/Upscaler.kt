package com.example.upscaler

import android.content.Context
import android.graphics.Bitmap
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

    fun upscale(bitmap: Bitmap): UpscaleResult {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val input = bitmapToFloatArray(resized)
        val output = FloatArray(outputWidth * outputHeight * 3)
        val elapsedMs = runInference(input, output)
        val result = floatArrayToBitmap(output, outputWidth, outputHeight)
        return UpscaleResult(result, elapsedMs)
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
