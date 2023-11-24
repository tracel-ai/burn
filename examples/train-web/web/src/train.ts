import wasm, { init, run } from 'train'
import workerUrl from './worker.ts?url'
import initSqlJs, { type Database } from 'sql.js'
import sqliteWasmUrl from './assets/sql-wasm.wasm?url'

// @ts-ignore https://github.com/rustwasm/console_error_panic_hook#errorstacktracelimit
Error.stackTraceLimit = 30

const sqlJs = initSqlJs({
	locateFile: () => sqliteWasmUrl,
})

// just load and run mnist on pageload
fetch('./mnist.db')
	.then((r) => r.arrayBuffer())
	.then(loadSqliteAndRun)
	.catch(console.error)

export function setupTrain(element: HTMLInputElement) {
	element.onchange = async function () {
		const files = element.files
		if (files != null) {
			const file = files[0]
			if (file != null) {
				let ab = await file.arrayBuffer()
				await loadSqliteAndRun(ab)
			}
		}
	}
}

async function loadSqliteAndRun(ab: ArrayBuffer) {
	await wasm()
	await init(workerUrl, navigator.hardwareConcurrency)
	await sleep(1000) // the workers need time to spin up. TODO, post an init message and await a response. Also maybe move worker construction to Javascript.
	// Images are an array of arrays.
	// We can't send an array of arrays to Wasm.
	// So instead we merge images into a single large array and
	// use another array, `lengths`, to keep track of the image size.
	// Encoding is done here.
	// Decoding is done at `burn/examples/train-web/train/src/mnist.rs`.
	const trainImages: Uint8Array[] = []
	const trainLabels: number[] = []
	const trainLengths: number[] = []
	const testImages: Uint8Array[] = []
	const testLabels: number[] = []
	const testLengths: number[] = []
	const db = await getDb(ab)
	try {
		const trainQuery = db.prepare('select label, image_bytes from train')
		while (trainQuery.step()) {
			const row = trainQuery.getAsObject()
			const label = row.label as number
			trainLabels.push(label)
			const bytes = row.image_bytes as Uint8Array
			trainImages.push(bytes)
			trainLengths.push(bytes.length)
		}
		trainQuery.free()
		const testQuery = db.prepare('select label, image_bytes from test')
		while (testQuery.step()) {
			const row = testQuery.getAsObject()
			const label = row.label as number
			testLabels.push(label)
			const bytes = row.image_bytes as Uint8Array
			testImages.push(bytes)
			testLengths.push(bytes.length)
		}
		testQuery.free()
	} finally {
		db.close()
	}
	run(
		Uint8Array.from(trainLabels),
		concat(trainImages),
		Uint16Array.from(trainLengths),
		Uint8Array.from(testLabels),
		concat(testImages),
		Uint16Array.from(testLengths),
	)
}

async function getDb(ab: ArrayBuffer): Promise<Database> {
	const sql = await sqlJs
	return new sql.Database(new Uint8Array(ab))
}

// https://stackoverflow.com/a/59902602
function concat(arrays: Uint8Array[]) {
	const totalLength = arrays.reduce((acc, value) => acc + value.length, 0)
	const result = new Uint8Array(totalLength)
	let length = 0
	for (const array of arrays) {
		result.set(array, length)
		length += array.length
	}
	return result
}

async function sleep(ms: number): Promise<unknown> {
	return await new Promise((resolve) => setTimeout(resolve, ms))
}
