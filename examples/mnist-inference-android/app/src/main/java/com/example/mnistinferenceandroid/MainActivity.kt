package com.example.mnistinferenceandroid

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.example.mnistinferenceandroid.ui.theme.MnistInferenceAndroidTheme

class MainActivity : ComponentActivity() {
    init {
        System.loadLibrary("mnist_inference_android")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MnistInferenceAndroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MnistRecognizePage()
                }
            }
        }
    }
}
