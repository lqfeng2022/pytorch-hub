import { Box, Flex, Image } from '@chakra-ui/react';
import book_img from '../../src/assets/book_cover.jpeg';

const CoverPage = () => {
  return (
    <Flex alignItems='center' justifyContent='center'>
      <Box maxW='500px'>
        <Image src={book_img} />
      </Box>
    </Flex>
  );
};

export default CoverPage;