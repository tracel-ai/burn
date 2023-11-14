import wasm, { init, run } from 'train'
import workerUrl from './worker.ts?url'
import initSqlJs, { type Database } from 'sql.js'
import sqliteWasmUrl from './assets/sql-wasm.wasm?url'

// @ts-ignore https://github.com/rustwasm/console_error_panic_hook#errorstacktracelimit
Error.stackTraceLimit = 30

wasm()
	.then(() => init(workerUrl))
	.catch(console.error)

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
	const imageBytes: Uint8Array[] = []
	const labels: number[] = []
	const lengths: number[] = []
	const db = await getDb(ab)
	try {
		const trainQuery = db.prepare('select label, image_bytes from train')
		while (trainQuery.step()) {
			const row = trainQuery.getAsObject()
			const label = row.label as number
			labels.push(label)
			const bytes = row.image_bytes as Uint8Array
			imageBytes.push(bytes)
			lengths.push(bytes.length)
		}
		trainQuery.free()
	} finally {
		db.close()
	}
	run(Uint8Array.from(labels), concat(imageBytes), Uint16Array.from(lengths))
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
