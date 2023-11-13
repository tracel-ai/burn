import wasm, { run } from 'train'
import workerUrl from './worker.ts?url'
import initSqlJs, { type Database } from 'sql.js'
import sqliteWasmUrl from './assets/sql-wasm.wasm?url'

// @ts-ignore https://github.com/rustwasm/console_error_panic_hook#errorstacktracelimit
Error.stackTraceLimit = 30

export function setupTrain(element: HTMLInputElement) {
	element.onchange = async function () {
		await wasm()
		const db = await getDb(element.files![0])
		const trainQuery = db.prepare('select label, image_bytes from train')
		const imageBytes: Uint8Array[] = []
		const labels: number[] = []
		const lengths: number[] = []
		while (trainQuery.step()) {
			const row = trainQuery.getAsObject()
			const label = row.label as number
			labels.push(label)
			const bytes = row.image_bytes as Uint8Array
			imageBytes.push(bytes)
			lengths.push(bytes.length)
		}
		run(
			workerUrl,
			Uint8Array.from(labels),
			concat(imageBytes),
			Uint16Array.from(lengths),
		)
	}
}

async function getDb(sqliteBuffer: File): Promise<Database> {
	const sql = await initSqlJs({
		locateFile: () => sqliteWasmUrl,
	})
	const buffer = await sqliteBuffer.arrayBuffer()
	return new sql.Database(new Uint8Array(buffer))
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
