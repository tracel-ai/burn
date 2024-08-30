package com.example.mnistinferenceandroid

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

external fun infer(inputImage: ByteArray): Int;
@Composable
fun MnistRecognizePage() {
    var result by remember { mutableStateOf<Int?>(null) }
    val context = LocalContext.current

    // Launcher to pick an image from the file explorer
    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val byteArray = uriToByteArray(context, it);
            if(byteArray != null){
                result = infer(byteArray)
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Button(onClick = { pickImageLauncher.launch("image/*") }) {
            Text("Select Image")
        }

        Spacer(modifier = Modifier.height(16.dp))

        result?.let {
            if (it == -1)
                Text(text = "Model prediction error")
            else
                Text("The number is: $it")
        }
    }
}