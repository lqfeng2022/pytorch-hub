import { Center, Box, Text, Stack } from '@chakra-ui/react'

const Definition = () => {
  const definitions = [
    { id: 0, 
      name: "Artificial Intelligence", 
      value: "Artificial Intelligence (AI) is a branch of Computer Science that makes Machines and Systems perform tasks that Simulate Human Intelligence.",
      contents: [
        { id: 0, 
          name: "Artificial Intelligence (AI) is a branch of computer science dedicated to developing machines and systems that can perform tasks traditionally requiring human intelligence. These tasks include learning from experience, adapting to new information, reasoning through complex problems, recognizing patterns, understanding and processing natural language, and even making decisions."
        },
        { id: 1, 
          name: "AI systems can be categorized into different levels, from narrow AI, which is designed for specific tasks like facial recognition or language translation, to general AI, which would have the capability to perform any intellectual task that a human can do. Although we are currently at the stage of narrow AI, advancements in the field are rapidly evolving, pushing the boundaries of what machines can achieve."
        },
        { id: 2,
          name: "AI technologies power many aspects of our daily lives, from recommendation algorithms on streaming services to virtual assistants like Siri and Alexa, autonomous vehicles, and sophisticated data analysis tools used in industries like healthcare, finance, and beyond. As AI continues to evolve, it holds the potential to transform industries, drive innovation, and address complex global challenges, but it also raises important ethical and societal considerations that need to be carefully managed."
        }
      ]
    }
  ]

  return (
    <>
      <Center my={8} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{definitions[0].value}</Text>
        </Box>
      </Center>
      <Stack my={5} spacing={4}>
        {definitions[0].contents.map((p) => <Text key={p.id}>{p.name}</Text>)}
      </Stack>
    </>
  )
}

export default Definition