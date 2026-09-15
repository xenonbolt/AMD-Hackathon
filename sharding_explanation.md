# Understanding Sharding (FSDP): A Beginner's Guide

If you've never dealt with distributed machine learning before, "Fully Sharded Data Parallel" (FSDP) can sound intimidating. Let's break it down using a simple analogy.

## The Problem: The Giant Encyclopedia
Imagine you have a giant, 10,000-page encyclopedia (our **AI Model**). You need to read it, find mistakes, and write corrections. 
However, this book is so heavy that a single person (your **Master Node**) cannot physically hold it without dropping it. In computer terms, the model is so large that it exceeds the server's RAM, causing an "Out of Memory" (OOM) crash.

## The Solution: Sharding (The Group Project)
Instead of forcing one person to hold the entire book, you gather a team of 4 friends (your **1 Master and 3 Worker Nodes**). You rip the encyclopedia into 4 equal chunks and give one chunk to each person. 
This process of ripping the book and distributing it is called **Sharding**. 

### What exactly was divided?
In our FSDP setup (`fsdp=["full_shard"]`), we divided three specific things across your 4 Ubuntu servers:
1. **Model Parameters (The Book Pages):** The actual "brain" or weights of the AI model.
2. **Gradients (The Sticky Notes):** When the AI makes a mistake during training, it calculates a gradient (a correction). We split the memory required to hold these corrections.
3. **Optimizer States (The Study Guide):** The optimizer (like AdamW) keeps a historical record of past corrections to make better future corrections. This takes up a massive amount of memory, so it is also divided.

By dividing all three, no single server has to hold more than 25% of the total memory burden!

---

## How It Works in Practice

You might wonder: *"If Server 1 only has Chapter 1, and Server 2 only has Chapter 2... how do they read the whole book?"*

They communicate over the network (using your **Gloo** backend). Here is the step-by-step process of how your cluster trains the model:

### Step 1: The Forward Pass (Making a Prediction)
Let's say the AI is currently processing the very first layer of its neural network (Layer 1).
- **Node A** owns Layer 1. 
- Node A rapidly sends copies of Layer 1 over the network to Node B, C, and D.
- Now, for a brief millisecond, all 4 nodes have Layer 1. They all do the math to make a prediction.
- As soon as they finish the math for Layer 1, **Nodes B, C, and D instantly delete their copies**. 
- They move on to Layer 2 (which Node B owns). Node B shares it, they do the math, and instantly delete it.

*Why delete it?* Because if they kept it, their RAM would fill up and crash. They only hold what they need, exactly when they need it.

### Step 2: The Backward Pass (Learning from Mistakes)
Once the model makes a prediction, it compares it to the correct answer to see how wrong it was. It needs to calculate the **Gradients** (the corrections) by going backward from the last layer to the first.
- The process repeats in reverse. The node that owns the last layer shares it, everyone calculates the corrections, and then they instantly throw the layer away.

### Step 3: The Update
Finally, the model needs to apply the corrections to get smarter.
- Because the parameters and optimizer states are sharded, **each node only updates its own 25% of the model**. 
- Node A updates its chunk. Node B updates its chunk.
- No massive memory spikes occur because nobody is trying to update the entire 100% at once.

---

## Why did we need `setup_swap.sh`?
In the very beginning, before the "book" is ripped up and shared, the PyTorch system briefly needs to load the entire model to figure out how to divide it. This initial loading phase creates a massive, temporary spike in memory. 

By creating a **16GB Swap file** on the hard drive of every node, we give the servers temporary "fake RAM" to survive that initial loading spike. Once the model is sharded and distributed, the RAM usage drops back down to safe levels, and the swap is barely used.
