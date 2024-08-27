import { 
  Box,
  Button,
  Center, 
  Image, 
  Stack, 
  Text,
  useBreakpointValue,
} from '@chakra-ui/react';
import book_img from '../src/assets/book_cover.jpeg';
import book_img_2 from '../src/assets/book_cover_2.jpeg';
import { useState } from 'react';
import BookList from './components/BookList';

function HomePage() {
  const [show, setShow] = useState(false)
  const bookImage = useBreakpointValue({ 
    base: book_img, lg: book_img_2 
  });

  const handleShow = () => { setShow(true)}
  const handleHide = () => { setShow(false)}

    // Determine if content should be visible based on screen size and state
  const isLargeScreen = useBreakpointValue({ base: false, lg: true });
  const shouldShowContent = isLargeScreen || !show;
  
  return (
    <Center>
      <Stack spacing={5} pb={5}>
        { shouldShowContent && <>
          <Image src={bookImage} alt='Book Cover'/>
          <Text>Hi! Hello! I built this web for this book. You can read the whole contents here. This book is a beginners' guider for anyone interested in Artificial Intelligence. If this book brings you inspiration and joy, and you want to dive deeper, feel free to visit my YouTube channel, where I'm gonna talk more details about <Text as='b'>Artificial Intelligence</Text> and <Text as='b'>Developments</Text>.</Text>
          {!isLargeScreen && 
            <Button 
              colorScheme='pink' 
              w='100px' 
              h='35px' 
              onClick={handleShow}
            >
              SHALL WE
            </Button>
          }
        </>}
        {useBreakpointValue({base: show, lg: false}) && 
          <Box 
            w={{ base: '350px', sm: '450px', md: '700px' }}
            position='relative'
            pb='70px'
          >
            <BookList/>
            <Button 
              colorScheme='gray' 
              w='80px' 
              h='35px' 
              onClick={handleHide}
              position='absolute'
              right='0'
              bottom='30px'
            >
              CLOSE
            </Button>
          </Box>
        }
      </Stack>
    </Center>
  );
}

export default HomePage;