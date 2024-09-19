import { 
  Box,
  Button,
  Center, 
  HStack, 
  Image, 
  Stack, 
  Text,
  useBreakpointValue,
} from '@chakra-ui/react';
import book_img from './assets/book_cover.jpeg';
import book_img_2 from './assets/book_cover_2.jpeg';
import { useState } from 'react';
import BookList from './components/BookList';

function HomePage() {
  const [show, setShow] = useState(false)
  const bookImage = useBreakpointValue({ 
    base: book_img, lg: book_img_2 
  });

  const isLargeScreen = useBreakpointValue({ base: false, lg: true });
  const showContent = isLargeScreen || !show;

  const handleShow = () => { setShow(true)}
  const handleHide = () => { setShow(false)}
  
  return (
    <Center>
      <Stack spacing={3} pb={5}>
        { showContent && <>
          <Text>
            What's up! I built this web for a PyTorch training course, and it's designed for beginners who're interested in Artificial Intelligence.
          </Text>
          <Text fontWeight='bold'>Let's hit the road!</Text>
          { !isLargeScreen && 
            <Button 
            bg='tomato'
            color='white'
            w='100px' 
            h='35px' 
            onClick={handleShow}
            _hover={{ bg: '#E53E3E' }}
            >SHALL WE</Button>
          }
          <Image src={bookImage} alt='Book Cover'/>
        </>}
        { useBreakpointValue({base: show, lg: false}) && 
          <Box 
            w={{ base: '350px', sm: '450px', md: '700px' }}
            position='relative'
            pb='70px'
          >
            <HStack justifyContent='space-between'>
              <Text fontWeight='bold' color='gray.500' borderBottom='1px solid red'>BOOK LIST:</Text>
              <Button
                colorScheme='gray'
                color='gray.500'
                w='80px'
                h='35px'
                onClick={handleHide}
              >CLOSE</Button>
            </HStack>
            <BookList/>
          </Box>
        }
      </Stack>
    </Center>
  );
}

export default HomePage;