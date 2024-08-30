# MNIST number detector Android App

This project is a sample Android application that demonstrates how to integrate a `Burn` into an android app using the JNI (Java Native Interface).

## Table of Contents

- [Workflow](#workflow)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [How To make your own](#how-to-make-your-own)
- [License](#license)

## Workflow
1. **Image Input:** The user provides an image input through the app's interface.
2. **Image Processing:** The image is converted to a grayscale `byteArray` in Kotlin.
3. **JNI Bridge:** The grayscale `byteArray` is passed to a Rust function via JNI.
4. **Rust Processing:** The Rust function calls the `forward` method from the `burn` library, using a pretrained MNIST ONNX model to perform inference.
5. **Result Handling:** The result, an integer representing the predicted digit, is logged to the android console and returned from Rust to Kotlin.
6. **Output Display:** The predicted digit is displayed on the screen

## Prerequisites
- Android Studio (latest version recommended)
- Rust (installed and configured)
- Android NDK (Native Development Kit)

## Setup
1. **Install Rust dependencies:**

    Ensure Rust is installed and the `cargo` command is available:

    ```bash
    rustup update
    ```
    And that you have installed all the rustup toolchains required: 
    ```bash
    rustup target add \
        aarch64-linux-android \
        armv7-linux-androideabi \
        i686-linux-android \
        x86_64-linux-android
    ```

2. **Configure the Android NDK:**

    Ensure that the Android NDK is installed. You can install it via Android Studio's SDK Manager.

3. **Build the android app:**

    Running the android app should automatically build the rust libraries due to the gradle tasks configured at the app level. (More on that later)


## How To make your own
1. There are a few ways to compile a rust library for android - 
    - Add targets in `.cargo/config.toml` and build with them. Then we can add the `.so` files generated to the jni directory in `app/src/main/jniLibs`
    - Add gradle plugins (like [rust-android-gradle](https://github.com/mozilla/rust-android-gradle) or [cargo-ndk-android](https://github.com/willir/cargo-ndk-android-gradle) using `rust-android-gradle` in this project) to do the work for you, so that the rust library is built on each app build. (Might want to change for expensive library builds)
2. To interface with Kotlin(Java) you can either use an interface generator (like [flapigen-rs](https://github.com/Dushistov/flapigen-rs)) or make them by yourself. This sample function doesn't use flapigen.
3. Now the function to be called from android (`infer()` here) needs to follow the [JNI naming conventions](https://docs.oracle.com/javase/1.5.0/docs/guide/jni/spec/design.html) (The correct name is also shown in the call error if it doesn't exist).
4. **Important** The first 2 arguments of the jni interfacing function will be the `env` variable (for interface functions) and the `this` object. The data you pass will start from the 3rd argument.
5. Next for converting the data from java to rust data types, there are multiple functions in the env variable passed to the function. Use as required...
6. Then in the app's `build.gradle` we add the part to run the cargo build before building the app and the also the cargo build details: 
```kotlin
// Cargo build details
cargo {
    module = "./src/main/rust"       // Or whatever directory contains your Cargo.toml
    libname = "mnist_inference_android"          // Or whatever matches Cargo.toml's [package] name.
    targets = listOf(
        "arm", "arm64",
        "x86",
        "x86_64"
    )
    prebuiltToolchains = true
}

// Used to build cargo before the android build task is run
// See more options here: https://github.com/mozilla/rust-android-gradle/issues/133
project.afterEvaluate {
    tasks.withType(com.nishtahir.CargoBuildTask::class)
        .forEach { buildTask ->
            tasks.withType(com.android.build.gradle.tasks.MergeSourceSetFolders::class)
                .configureEach {
                    this.inputs.dir(
                        layout.buildDirectory.dir("rustJniLibs" + File.separatorChar + buildTask.toolchain!!.folder)
                    )
                    this.dependsOn(buildTask)
                }
        }
}
```
(In the example we have also added the target directory in `config.toml` since otherwise it will build into the workspace target, which we do not want)
7. Here the library's name is `mnist-android` so we will initialize it in our app:
```kotlin
class MainActivity : ComponentActivity() {
    init {
        System.loadLibrary("mnist_android")     // Note: '-' is changed to '_'
    }
    ...
}
```
8. Finally use it by declaring it as an external function first
```kotlin
external fun infer(inputImage: ByteArray): Int;

...
infer(byteArray)
```