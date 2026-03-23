from langchain_text_splitters import RecursiveCharacterTextSplitter


SAMPLE_CODE = """// Superclass (Parent Class)

class Animal {
    String name;

    public void eat() {
        System.out.println("I can eat");
    }
}

// Subclass (Child Class) that inherits from Animal
class Dog extends Animal {
    // New method in the subclass
    public void display() {
        // Accessing the field of the superclass
        System.out.println("My name is " + name);
    }

    // Method overriding (optional, but a common use of inheritance/polymorphism)
    @Override
    public void eat() {
        System.out.println("The dog eats food");
    }
}

// Main class to test the inheritance
public class Main {
    public static void main(String[] args) {
        // Create an object of the subclass
        Dog labrador = new Dog();

        // Access field of superclass using subclass object
        labrador.name = "Rohu";
        
        // Call method of subclass
        labrador.display(); 

        // Call overridden method (first looks in Dog, then in Animal)
        labrador.eat();
    }
}
"""

textSplitter = RecursiveCharacterTextSplitter.from_language("java", chunk_size=200, chunk_overlap=0)

chunks = textSplitter.create_documents([SAMPLE_CODE])

print(f"Original code length: {len(SAMPLE_CODE)} characters")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")