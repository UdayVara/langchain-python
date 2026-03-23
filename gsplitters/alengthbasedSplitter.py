from langchain_text_splitters import RecursiveCharacterTextSplitter 

sample_text = """On lazy days in summer there is cricket,
At the Oval where the greatest are at play.

I watch the action from behind the wicket
And slowly let the hours while away.

 

Of course it's just as likely I'll be dozing,

And changing what I see for what I dream.
Quite often I shall fancy I'm imposing
My copious talents on the other team.

 

Exactly which is which will hardly matter,
It's being there with friends (or on my own).
Some tea and scones and then a friendly natter
The trials of life are briefly overthrown.

 

All Winter life's a trial but in my mind
Those Oval idylls can't be far behind!"""


textSplitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)

chunks = textSplitter.create_documents([sample_text])

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}\n")