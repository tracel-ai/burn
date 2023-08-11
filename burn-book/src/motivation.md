# Why Burn?


Why bother with the effort of creating an entirely new deep learning framework from scratch when PyTorch, TensorFlow, and other frameworks already exist?
Spoiler alert: Burn isn't merely a replication of PyTorch or TensorFlow in Rust. 
It represents a novel approach, placing significant emphasis on making the right compromises in the right areas to facilitate exceptional flexibility, high performance, and a seamless developer experience.
Burn isn’t a framework specialized for only one type of application, it is designed to serve as a versatile framework suitable for a wide range of research and production uses.
The foundation of Burn's design revolves around three key user profiles:

**Machine Learning Researchers** require tools to construct and execute experiments efficiently.
It’s essential for them to iterate quickly on their ideas and design testable experiments which can help them discover new findings.
The framework should facilitate the swift implementation of cutting-edge research while ensuring fast execution for testing.

**Machine Learning Engineers** are another important demographic to keep in mind.
Their focus leans less on swift implementation and more on establishing robustness, seamless deployment, and cost-effective operations.
They seek dependable, economical models capable of achieving objectives without excessive expense.
The whole machine learning workflow —from training to inference— must be as efficient as possible with minimal unpredictable behavior.

**Low level Software Engineers** working with hardware vendors want their processing units to run models as fast as possible to gain competitive advantage.
This endeavor involves harnessing hardware-specific features such as Tensor Core for Nvidia.
Since they are mostly working at a system level, they want to have absolute control over how the computation will be executed.

The goal of Burn is to satisfy all of those personas!
