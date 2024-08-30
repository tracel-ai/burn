package com.example.mnistinferenceandroid

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri

fun uriToByteArray(context: Context, uri: Uri): ByteArray? {
    val inputStream = context.contentResolver.openInputStream(uri) ?: return null
    val byteArray = inputStream.readBytes()
    val imageMap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)

    // The model takes 28x28 images as input so reduce size before grayscale conversion
    val reducedMap = Bitmap.createScaledBitmap(imageMap, 28, 28, false)

    val pixelArray = convertToGrayscaleArray(reducedMap)
    return pixelArray
}

fun convertToGrayscaleArray(bmp: Bitmap): ByteArray {
    // Create a mutable bitmap with the same dimensions as the original
    val width = bmp.width
    val height = bmp.height

    val grayscaleArray = ByteArray(width * height)
    // Iterate over each pixel in the original bitmap
    for (y in 0 until height) {
        for (x in 0 until width) {
            // Get the pixel color at (x, y)
            val pixel = bmp.getPixel(x, y)

            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            // Converting to grayscale using the NTSC formula
            val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            grayscaleArray[x + y * width] = gray.toByte() // Can also use the int array directly
        }
    }
    return grayscaleArray
}
