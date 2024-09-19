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
import { Link } from 'react-router-dom';

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
            What's up! I built this site for a PyTorch Model Building course, and it's perfect for beginners interested in Artificial Intelligence.
          </Text>
          <HStack justifyContent='space-between'>
            { !isLargeScreen &&
              <Text 
                fontSize='xl' 
                color='gray.500' 
                borderBottom='1px solid #FEB2B2'
                onClick={handleShow}
                _hover={{ 
                  color: 'tomato', 
                  cursor: 'pointer',
                  transform: 'translateX(5px)',
                  transition: 'transform .15s ease-in',
                }}
              >
                Let's hit the road:
              </Text>
            }
          </HStack>
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
                _hover={{ bg: '#FF6347', color: '#FFFFFF' }}
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