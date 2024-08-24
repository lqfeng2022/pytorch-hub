import { Box, Stack, Text, Image } from '@chakra-ui/react'
import dlapps from '../../assets/dlapps.jpeg'

const DLApps = () => {
  const comparing = {
    name: "3.3 Deep Learning Applications",
    image: dlapps,
    values: [
      { id: 0, 
        title: "1) Image Recognition",
        content: "Image recognition is a part of computer vision that helps computers figure out what's in a picture or camera. Whether it's a photo of your dog, a face, or a stop sign. Image recognition lets the computer look at the image, analyze it, and then tell you what it sees. Image recognition is everywhere, like when you pick up your iPhone, you gotta use it to recognizes your face to unlock, like helping self-driving or autopilot cars see and avoid obstacles on the road."
      },
      { id: 1, 
        title: "2) Natural Language Processing (NLP)", 
        content: "Natural language processing or NLP that helps computers understand and interact with human language, it's all about making computers smart enough to read, write, and talk in a way that makes sense to us. NLP is used in a lot of cool ways, for example, it can translate languages, like Google Translate does, turn speech to text, like when you talk to Siri, and generate natural language text, such as writing stories and articles, like ChatGPT generating text. In short, NLP is what makes it possible for computers to communicate with us in a natural way, whether that's through text or speech."
      },
      { id: 2, 
        title: "3) Speech Recognition", 
        content: "Speech recognition is a technology that enables computers to understand and process spoken language. It's what allows your phone, smart speakers, and other devices to “listen” to what you say and response accordingly. Speech recognition is what powers voice assistant like Siri, allowing you to interact with your devices just by talking to them."
      },
      { id: 3, 
        title: "4) A Recommendation System",
        content: "A recommendation is a type of technology used to suggest products, services, or content to users based on their preferences, behaviors, or other data. It's what helps online platforms like Amazon, JD show you things you might like. In everyday life, recommendation systems are all around you, helping you discover new products, movies, music, or even friends on social media. They're designed to make your experience more personalized and relevant by showing you things that match your tastes and interests. However, it can also come with risks like privacy issues and the potential to limit what we see (the filter bubble). "
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{comparing.name}</Text>
      <Image py={5} src={comparing.image}/>
      <Stack spacing={4}>
        {comparing.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text>{p.content}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default DLApps