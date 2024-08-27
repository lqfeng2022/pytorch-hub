import { 
  Button,
  Center, 
  Image, 
  Stack, 
  Text, 
  useBreakpointValue 
} from '@chakra-ui/react';
import book_img from '../src/assets/book_cover.jpeg';
import book_img_2 from '../src/assets/book_cover_2.jpeg';

function HomePage() {
  // Determine which image to show based on the current screen size
  const bookImage = useBreakpointValue({ 
    base: book_img, lg: book_img_2 
  });
  
  return (
    <Center>
      <Stack spacing={5}>
        <Image src={bookImage} alt='Book Cover' />
        <Text>Hi! Hello! I built this web for this book. You can read the whole contents here. This book is a beginners' guider for anyone interested in Artificial Intelligence. If this book brings you inspiration and joy, and you want to dive deeper, feel free to visit my YouTube channel, where I'm gonna talk more details about Artificial Intelligence, Developments, and even Quantum Computing Science.</Text>
        <Button colorScheme='pink' w='100px' h='35px'>SHALL WE</Button>
      </Stack>
    </Center>
  );
}

export default HomePage;