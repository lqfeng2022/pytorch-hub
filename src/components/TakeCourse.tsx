import { Box, Stack, Text, Image } from '@chakra-ui/react'
import takecourse_img from '../assets/takecourse_img.jpeg'

const TakeCourse = () => {
  const takeCourse = {
    title: "8. How to take this course?",
    subtitle: "What should we do before this trainig course, first of all, you should know Python programming languages. Then you gotta know some mathematic knowledges, like Matrix computation, calculus, probability and statistics.",
    image: takecourse_img,
    values: [
      { id: 1, 
        title: "Watch all lessons from A to Z",
        value: "First and for most, I want you to watch this entire course all the way from beginning to the end, even you're familiar with PyTorch. Cus along the way, I am gonna share lots of tips and tricks for understanding, and I don't want you to miss out on any of them. And in this course, I am not gonna wast your time on repetitive and useless stuff. So make sure to watch all lessons."
      },
      { 
        id: 2, 
        title: "Take notes",
        value: "Now while watching the lesson, I want you to take notes. You can just write down some keywords on your paper, if you don't wanna take lots of notes. I strongly believed that writing things down will help you remember new things that you've learned. Then after each lesson, go through your notes and repeat the same steps I showed to you. This is exactly how I personally learn new things."
      },
      { id: 3, 
        title: "Complete all exercises",
        value: "Also, I've carefully designed some exercises that help you understand and remember what we learned. So make sure to do all the practices. Cus the more you practice, the better you will be in machine learning field in general."
      },
      { id: 4, 
        title: "Share your work",
        value: "The more you share, the more you get. If you wanna a deeper comprehension about this course, the best  way is building your own project with your personal understanding, then share it.  When you share your work, the more people will get to know you and give you feedback on your work. So be open to feedback, listen to it with an open mind. It can help you improve your skills and create better work in the future. I believe you can do better than me."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{takeCourse.title}</Text>
      <Text fontSize='lg' pt={3}>{takeCourse.subtitle}</Text>
      <Image py={5} src={takeCourse.image}/>
      <Stack spacing={4}>
        {takeCourse.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text py={2}>{p.value}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default TakeCourse