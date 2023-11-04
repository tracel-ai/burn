import wasm, { run } from 'train'

Error.stackTraceLimit = 30

await wasm()

export function setupCounter(element: HTMLButtonElement) {
	let counter = 0
	const setCounter = (count: number) => {
		run()
		counter = count
		element.innerHTML = `count is ${counter}`
	}
	element.addEventListener('click', () => setCounter(counter + 1))
	setCounter(0)
}
