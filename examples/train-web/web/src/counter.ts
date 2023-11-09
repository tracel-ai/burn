import wasm, { run } from 'train'
import workerUrl from './worker.ts?url'

// @ts-ignore https://github.com/rustwasm/console_error_panic_hook#errorstacktracelimit
Error.stackTraceLimit = 30

await wasm()
run(workerUrl)

export function setupCounter(element: HTMLButtonElement) {
	let counter = 0
	const setCounter = (count: number) => {
		counter = count
		element.innerHTML = `count is ${counter}`
	}
	element.addEventListener('click', () => setCounter(counter + 1))
	setCounter(0)
}
